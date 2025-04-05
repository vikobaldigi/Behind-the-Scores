# %%
import json
import pandas as pd

# Path to the GeoJSON file
geojson_path = '/home/vkobal/Documents/NYC_ZIP_Code_View.geojson'

# Load the GeoJSON file
with open(geojson_path, 'r') as f:
    data = json.load(f)

# Print the basic structure
print("Type of GeoJSON:", type(data))
print("\nKeys in the GeoJSON:", data.keys())

# Check the type of features
print("\nNumber of features:", len(data['features']))
print("\nExample of first feature:")
print(json.dumps(data['features'][0], indent=2)[:1000] + "...")  # Truncated for readability

# Examine the properties available in features
if len(data['features']) > 0:
    print("\nProperties in first feature:")
    properties = data['features'][0]['properties']
    for key, value in properties.items():
        print(f"- {key}: {type(value).__name__} example: {value}")

# Extract ZIP codes and create a summary dataframe
zip_codes = []
for feature in data['features']:
    props = feature['properties']
    # Assuming there's a property for ZIP code - adjust the key name if needed
    # Common keys might be 'ZIPCODE', 'ZIP', 'postalCode', etc.
    zip_info = {
        'geometry_type': feature['geometry']['type']
    }
    
    # Add all properties to our info dictionary
    for key, value in props.items():
        zip_info[key] = value
    
    zip_codes.append(zip_info)

# Create a DataFrame
zip_df = pd.DataFrame(zip_codes)
print("\nDataFrame columns:", zip_df.columns.tolist())
print("\nDataFrame info:")
print(zip_df.info())
print("\nFirst few rows:")
print(zip_df.head())

# Count of geometry types
print("\nGeometry types count:")
print(zip_df['geometry_type'].value_counts())

# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import os

# Base directory
base_dir = '/home/vkobal/Documents/'

# File paths
geojson_path = os.path.join(base_dir, 'NYC_ZIP_Code_View.geojson')
output_map_path = os.path.join(base_dir, 'nyc_zip_code_map.png')
output_choropleth_path = os.path.join(base_dir, 'nyc_zip_code_choropleth.png')
output_interactive_path = os.path.join(base_dir, 'nyc_zip_code_interactive_map.html')

# Read the GeoJSON file into a GeoDataFrame
nyc_zip_gdf = gpd.read_file(geojson_path)

# Print basic information
print(f"Total ZIP codes: {len(nyc_zip_gdf)}")
print(f"Counties included: {nyc_zip_gdf['COUNTY'].unique()}")
print(f"Original CRS: {nyc_zip_gdf.crs}")

# Reproject to a projected CRS appropriate for New York (NAD83 / New York Long Island - EPSG:2263)
# This will fix the area calculation warnings
nyc_zip_gdf = nyc_zip_gdf.to_crs(epsg=2263)
print(f"Projected CRS: {nyc_zip_gdf.crs}")

# Create a color map based on counties
counties = nyc_zip_gdf['COUNTY'].unique()
county_colors = plt.cm.tab10(np.linspace(0, 1, len(counties)))
county_color_dict = dict(zip(counties, county_colors))

# Create a figure and axis for the county-colored map
fig, ax = plt.subplots(figsize=(15, 15))

# Plot each county with a different color
for county, color in county_color_dict.items():
    county_gdf = nyc_zip_gdf[nyc_zip_gdf['COUNTY'] == county]
    county_gdf.plot(
        ax=ax,
        color=color,
        edgecolor='black',
        linewidth=0.3,
        alpha=0.7
    )

# Calculate areas correctly now that we have a projected CRS
nyc_zip_gdf['area'] = nyc_zip_gdf.geometry.area / 1000000  # Convert to square km for easier interpretation

# Add ZIP code labels (for larger areas only)
for idx, row in nyc_zip_gdf.iterrows():
    # Only label larger areas to avoid overcrowding
    if row['area'] > nyc_zip_gdf['area'].quantile(0.75):
        centroid = row.geometry.centroid
        ax.text(
            centroid.x, 
            centroid.y, 
            row['ZIP_CODE'],
            fontsize=8,
            ha='center',
            va='center',
            color='black',
            fontweight='bold'
        )

# Create a legend for counties
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
           markersize=10, label=county) for county, color in county_color_dict.items()
]
ax.legend(handles=legend_elements, title='County', loc='lower right')

# Set title and labels
plt.title('New York City ZIP Code Boundaries', fontsize=16)
plt.tight_layout()

# Add county names as annotations
county_centroids = nyc_zip_gdf.dissolve(by='COUNTY').centroid
for county, point in zip(county_centroids.index, county_centroids):
    plt.annotate(
        county,
        xy=(point.x, point.y),
        xytext=(point.x, point.y),
        fontsize=12,
        fontweight='bold',
        ha='center',
        va='center',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7)
    )

# Remove axis labels as they're not needed for a map
ax.set_axis_off()

# Save the map as a high-resolution image
plt.savefig(output_map_path, dpi=300, bbox_inches='tight')
print(f"Static map saved to {output_map_path}")

# Show interactive plot (comment out if running in a non-interactive environment)
# plt.show()

# Create a choropleth map demonstrating ZIP code statistics
# In this example, we'll use ZIP code area as the metric
# In your real analysis, you would replace this with educational metrics
fig2, ax2 = plt.subplots(figsize=(15, 15))

# Generate random data for demonstration (replace with real educational data)
nyc_zip_gdf['random_metric'] = np.random.uniform(0, 100, size=len(nyc_zip_gdf))

# Plot the choropleth map
nyc_zip_gdf.plot(
    column='random_metric',
    ax=ax2,
    cmap='viridis',
    edgecolor='black',
    linewidth=0.3,
    legend=True,
    legend_kwds={
        'label': "Example Metric",
        'orientation': "horizontal",
        'shrink': 0.6
    }
)

# Add county labels
for county, point in zip(county_centroids.index, county_centroids):
    plt.annotate(
        county,
        xy=(point.x, point.y),
        xytext=(point.x, point.y),
        fontsize=12,
        fontweight='bold',
        ha='center',
        va='center',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7)
    )

# Set title
plt.title('NYC ZIP Codes - Example Choropleth (Replace with Educational Data)', fontsize=16)
ax2.set_axis_off()

# Save the choropleth map
plt.savefig(output_choropleth_path, dpi=300, bbox_inches='tight')
print(f"Choropleth map saved to {output_choropleth_path}")

# Show the second plot (comment out if running in a non-interactive environment)
# plt.show()


# Interactive Map Version with Folium
def create_interactive_map():
    import folium
    
    # We need to convert back to geographic coordinates for Folium
    nyc_zip_gdf_wgs84 = nyc_zip_gdf.to_crs(epsg=4326)
    
    # Center coordinates for NYC
    nyc_center = [40.7128, -74.0060]
    
    # Create a base map
    m = folium.Map(
        location=nyc_center,
        zoom_start=10,
        tiles='CartoDB positron'
    )
    
    # Function to style the ZIP code areas
    def style_function(feature):
        county = feature['properties']['COUNTY']
        return {
            'fillColor': county_hex_colors.get(county, '#FFFFFF'),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7
        }
    
    # Function for tooltip
    def tooltip_function(feature):
        props = feature['properties']
        return folium.Tooltip(
            f"ZIP: {props['ZIP_CODE']}<br>"
            f"Area: {props['PO_NAME']}<br>"
            f"County: {props['COUNTY']}"
        )
    
    # Convert the RGB colors to hex for folium
    county_hex_colors = {
        county: colors.rgb2hex(color) 
        for county, color in county_color_dict.items()
    }
    
    # Add the GeoJSON data to the map
    folium.GeoJson(
        nyc_zip_gdf_wgs84.__geo_interface__,
        style_function=style_function,
        tooltip=tooltip_function,
    ).add_to(m)
    
    # Add a legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 150px; height: 140px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white; padding: 10px; border-radius: 5px;">
      <span style="font-weight: bold;">Counties</span><br>
    '''
    
    for county, color_hex in county_hex_colors.items():
        legend_html += f'''
        <i style="background: {color_hex}; width: 15px; height: 15px; 
                 display: inline-block; margin-right: 5px;"></i>
        {county}<br>
        '''
    
    legend_html += '</div>'
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save the map
    m.save(output_interactive_path)
    print(f"Interactive map saved to {output_interactive_path}")
    
    return m

# Create the interactive map
interactive_map = create_interactive_map()


# Example function to incorporate educational data
def educational_data_example():
    """
    This function demonstrates how you could incorporate educational data
    for your research on educational disparities.
    """
    print("\nExample of how to incorporate educational data:")
    print("-----------------------------------------------")
    
    # Create a sample educational dataset (replace with your actual data)
    # Let's simulate graduation rates per ZIP code
    sample_education_data = {
        'ZIP_CODE': nyc_zip_gdf['ZIP_CODE'].tolist(),
        'graduation_rate': np.random.uniform(65, 95, size=len(nyc_zip_gdf)),
        'college_attendance': np.random.uniform(40, 85, size=len(nyc_zip_gdf)),
        'math_scores': np.random.uniform(650, 800, size=len(nyc_zip_gdf)),
    }
    education_df = pd.DataFrame(sample_education_data)
    
    # In real usage, you would load your data from a CSV file like this:
    # education_df = pd.read_csv(os.path.join(base_dir, 'your_education_data.csv'))
    
    # Join the educational data with the geographic data
    nyc_education_gdf = nyc_zip_gdf.merge(education_df, on='ZIP_CODE', how='left')
    
    # Create a new figure for the educational data visualization
    fig3, ax3 = plt.subplots(figsize=(15, 15))
    
    # Create a choropleth map of graduation rates
    nyc_education_gdf.plot(
        column='graduation_rate',
        ax=ax3,
        cmap='RdYlBu',  # Red-Yellow-Blue colormap (low to high)
        edgecolor='black',
        linewidth=0.3,
        legend=True,
        legend_kwds={
            'label': "Graduation Rate (%)",
            'orientation': "horizontal",
            'shrink': 0.6
        }
    )
    
    # Add county labels
    county_centroids = nyc_education_gdf.dissolve(by='COUNTY').centroid
    for county, point in zip(county_centroids.index, county_centroids):
        plt.annotate(
            county,
            xy=(point.x, point.y),
            xytext=(point.x, point.y),
            fontsize=12,
            fontweight='bold',
            ha='center',
            va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7)
        )
    
    # Set title and format
    plt.title('NYC ZIP Codes - High School Graduation Rates (Sample Data)', fontsize=16)
    ax3.set_axis_off()
    
    # Save the educational data map
    education_map_path = os.path.join(base_dir, 'nyc_education_graduation_rates.png')
    plt.savefig(education_map_path, dpi=300, bbox_inches='tight')
    print(f"Sample educational data map saved to {education_map_path}")
    
    # You could create multiple maps for different metrics
    # Let's create another one for college attendance rates
    
    fig4, ax4 = plt.subplots(figsize=(15, 15))
    
    nyc_education_gdf.plot(
        column='college_attendance',
        ax=ax4,
        cmap='PuRd',  # Purple-Red colormap
        edgecolor='black',
        linewidth=0.3,
        legend=True,
        legend_kwds={
            'label': "College Attendance Rate (%)",
            'orientation': "horizontal",
            'shrink': 0.6
        }
    )
    
    # Add county labels
    for county, point in zip(county_centroids.index, county_centroids):
        plt.annotate(
            county,
            xy=(point.x, point.y),
            xytext=(point.x, point.y),
            fontsize=12,
            fontweight='bold',
            ha='center',
            va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7)
        )
    
    plt.title('NYC ZIP Codes - College Attendance Rates (Sample Data)', fontsize=16)
    ax4.set_axis_off()
    
    college_map_path = os.path.join(base_dir, 'nyc_education_college_rates.png')
    plt.savefig(college_map_path, dpi=300, bbox_inches='tight')
    print(f"Sample college attendance map saved to {college_map_path}")
    
    # Return the merged dataframe in case you want to use it for further analysis
    return nyc_education_gdf

# Run the educational data example (comment out if not needed)
education_gdf = educational_data_example()

print("\nAll maps created successfully!")
print("You can now view the generated map images in your Documents folder.")


