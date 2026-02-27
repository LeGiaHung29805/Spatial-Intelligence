import geopandas as gpd
import rasterio # Thư viện chuyên dùng để đọc file ảnh vệ tinh/DEM (Raster)

def overlay_dem():
    # 1. Nạp dữ liệu nhà đã sạch và file ảnh DEM 12.5m
    buildings = gpd.read_file("data/processed/buildings_cleaned.geojson")
    dem_path = "data/raw/dem/bat_xat_12.5m.tif"
    
    # 2. Mở file ảnh DEM bằng rasterio
    with rasterio.open(dem_path) as dem:
        # 3. KIỂM TRA HỆ TỌA ĐỘ (CRS): 
        # Nếu nhà dùng độ (WGS84) mà DEM dùng mét (UTM), chúng sẽ bị lệch.
        # Dòng này ép chúng phải khớp nhau hoàn toàn.
        if buildings.crs != dem.crs:
            buildings = buildings.to_crs(dem.crs)
            
        # 4. TÍNH TÂM NHÀ (Centroid):
        # Vì ngôi nhà là một đa giác, chúng ta lấy điểm chính giữa mái nhà 
        # để "chọc" xuống bản đồ cao độ lấy giá trị Z.
        coords = [(geom.centroid.x, geom.centroid.y) for geom in buildings.geometry]
        
        # 5. TRÍCH XUẤT ĐỘ CAO (Sample):
        # dem.sample(coords) sẽ nhìn vào từng pixel tại đúng vị trí tâm nhà
        # và lấy ra con số độ cao (ví dụ: 150.5m) gán vào cột 'elevation_z'.
        buildings['elevation_z'] = [val[0] for val in dem.sample(coords)]
    
    # 6. Lưu file kết quả cuối cùng
    buildings.to_file("data/processed/buildings_with_z.geojson", driver='GeoJSON')
    print("✅ Đã gán cao trình thành công.")