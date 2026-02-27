import geopandas as gpd 

def clean_data():

    raw_gee_path = "data/shapefiles/infrastructure/BatXat_Buildings_GEE_Fixed.shp"
    output_path = "data/processed/buildings_cleaned.geojson"
    
    
    gdf = gpd.read_file(raw_gee_path)
    
    
    gdf = gdf[['geometry', 'area_in_meters', 'confidence']]
    
    
    gdf.to_file(output_path, driver='GeoJSON')
    print(f"Đã làm sạch dữ liệu.")