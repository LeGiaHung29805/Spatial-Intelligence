import geopandas as gpd
import rasterio 

def overlay_dem():
    buildings = gpd.read_file("data/processed/buildings_cleaned.geojson")
    dem_path = "data/raw/dem/bat_xat_12.5m.tif"
    
   
    with rasterio.open(dem_path) as dem:

        if buildings.crs != dem.crs:
            buildings = buildings.to_crs(dem.crs)
            

        coords = [(geom.centroid.x, geom.centroid.y) for geom in buildings.geometry]
        
        
        buildings['elevation_z'] = [val[0] for val in dem.sample(coords)]
    

    buildings.to_file("data/processed/buildings_with_z.geojson", driver='GeoJSON')
    print("Đã gán cao trình thành công.")