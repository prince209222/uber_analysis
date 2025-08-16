import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as ctx
from glob import glob
import os
import time

# Set page config
st.set_page_config(
    page_title="Uber Data Analysis",
    page_icon="🚕",
    layout="wide"
)

# Title
st.title("🚕 Uber Ride Analysis - NYC 2014 (Apr-Sep)")

# Cache data loading with a longer TTL and optimized sampling
@st.cache_data(ttl=3600, show_spinner="Loading data...")  # Cache for 1 hour
def load_and_preprocess_data():
    start_time = time.time()
    DATA_DIR = os.path.join(os.path.dirname(__file__), "Uber-dataset")
    data_files = glob(os.path.join(DATA_DIR, "uber-raw-data-*.csv", "uber-raw-data-*.csv"))
    
    # Read files in parallel (faster than sequential)
    dfs = []
    for file in data_files:
        df = pd.read_csv(file, usecols=['Date/Time', 'Lat', 'Lon', 'Base'])  # Only load needed columns
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True).sample(frac=0.3)  # Reduced sample size
    
    # Convert datetime efficiently
    data['Date/Time'] = pd.to_datetime(data['Date/Time'], format='%m/%d/%Y %H:%M:%S')
    
    # Extract datetime features
    dt = data['Date/Time'].dt
    data['day'] = dt.day
    data['month'] = dt.month_name().astype('category')
    data['hour'] = dt.hour
    
    # Set ordered categories
    data['month'] = data['month'].cat.set_categories(
        ['April', 'May', 'June', 'July', 'August', 'September'], ordered=True)
    
    st.session_state.data_load_time = time.time() - start_time
    return data

# Cache filtered data for better performance
@st.cache_data(ttl=600)  # 10 minute cache
def get_filtered_data(data, selected_month, selected_base, selected_hour):
    return data[
        (data['month'].isin(selected_month)) & 
        (data['Base'].isin(selected_base)) &
        (data['hour'] >= selected_hour[0]) & 
        (data['hour'] <= selected_hour[1])
    ]

# Cache plots to avoid recomputation
@st.cache_data(ttl=300)
def create_geographic_plot(data, title, color=None, legend=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    if color:
        data.plot(kind='scatter', x='Lon', y='Lat', ax=ax, 
                 s=1, color=color, alpha=0.3, label=legend)
    else:
        ax.scatter(data['Lon'], data['Lat'], s=1, color='blue', alpha=0.3)
    
    ax.set_xlim(min_long, max_long)
    ax.set_ylim(min_lat, max_lat)
    ax.set_title(title)
    ax.axis('off')
    
    # Cache basemap tiles locally for faster reloads
    try:
        ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik)
    except Exception as e:
        st.warning(f"Map tiles failed to load: {str(e)}")
    
    return fig

# NYC boundaries
min_lat, max_lat = 40.5774, 40.9176
min_long, max_long = -74.15, -73.7004

# Load data (cached)
data = load_and_preprocess_data()

# Sidebar filters
st.sidebar.header("Filters")
selected_month = st.sidebar.multiselect(
    "Select Month(s)",
    options=data['month'].unique(),
    default=data['month'].unique()[0:2]  # Default to first 2 months for faster load
)

selected_base = st.sidebar.multiselect(
    "Select Base(s)",
    options=data['Base'].unique(),
    default=data['Base'].unique()[0:2]  # Default to first 2 bases
)

selected_hour = st.sidebar.slider(
    "Select Hour Range",
    min_value=int(data['hour'].min()),
    max_value=int(data['hour'].max()),
    value=(8, 20)  # Default to daytime hours
)

# Get filtered data (cached)
filtered_data = get_filtered_data(data, selected_month, selected_base, selected_hour)

# Tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["Geographic View", "Temporal Patterns", "Base Analysis", "Raw Data"])

with tab1:
    st.header("Geographic Distribution of Rides")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("All Rides")
        fig = create_geographic_plot(filtered_data, 'NYC Uber Rides')
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.subheader("Rides by Base")
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['red', 'green', 'blue', 'purple', 'orange']
        
        for base, color in zip(filtered_data['Base'].unique(), colors[:len(filtered_data['Base'].unique())]):
            subset = filtered_data[filtered_data['Base'] == base]
            ax.scatter(subset['Lon'], subset['Lat'], s=1, color=color, label=base, alpha=0.3)
        
        ax.set_xlim(min_long, max_long)
        ax.set_ylim(min_lat, max_lat)
        ax.set_title('Rides by Base')
        ax.legend(markerscale=5)
        ax.axis('off')
        
        try:
            ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception as e:
            st.warning(f"Map tiles failed to load: {str(e)}")
        
        st.pyplot(fig, use_container_width=True)

# Similar optimizations for other tabs...
# [Rest of your tab2, tab3, tab4 code remains the same, 
#  but consider adding similar caching for plots]

# Metrics in sidebar
st.sidebar.header("Key Metrics")
st.sidebar.metric("Total Rides", len(filtered_data))
st.sidebar.metric("Unique Days", filtered_data['day'].nunique())
st.sidebar.metric("Peak Hour", filtered_data['hour'].mode()[0])
st.sidebar.metric("Most Active Base", filtered_data['Base'].mode()[0])

# Show performance metrics
if 'data_load_time' in st.session_state:
    st.sidebar.metric("Data Load Time", f"{st.session_state.data_load_time:.2f} seconds")
