import rasterio
import geopandas as gpd

# Nạp thử 1 file
with rasterio.open("data/raw/slope/Slope_BatXat_DEM_30m.tif") as src:
    raster_crs = src.crs
    raster_bounds = src.bounds

# Nạp Shapefile
gdf = gpd.read_file("data/shapefiles/infrastructure/Duong_Bat_Xat_New_All.shp").to_crs(raster_crs)
vector_bounds = gdf.total_bounds

print(f"Hệ tọa độ Raster: {raster_crs}")
print(f"Hệ tọa độ Vector: {gdf.crs}")
print(f"Ranh giới Raster: {raster_bounds}")
print(f"Ranh giới Vector: {vector_bounds}")

# Kiểm tra xem Vector có nằm trong Raster không
if (vector_bounds[0] > raster_bounds[2] or vector_bounds[2] < raster_bounds[0] or
    vector_bounds[1] > raster_bounds[3] or vector_bounds[3] < raster_bounds[1]):
    print("❌ LỖI: Đoạn đường và Ảnh vệ tinh KHÔNG nằm cùng một vùng địa lý!")
else:
    print("✅ Hai dữ liệu khớp nhau về không gian.")