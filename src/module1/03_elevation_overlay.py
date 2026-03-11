import geopandas as gpd
import rasterio
import os

def overlay_dem():
    # 1. Cấu hình đường dẫn (Đảm bảo tên file DEM khớp với file bạn đã tải)
    input_path = "data/processed/buildings_cleaned.geojson"
    dem_path = "data/raw/dem/Copernicus_DEM_BatXat_30m.tif"
    output_path = "data/processed/buildings_with_z.geojson"
    
    if not os.path.exists(input_path):
        print(f"Không tìm thấy file nhà đã làm sạch tại: {input_path}")
        return
    if not os.path.exists(dem_path):
        print(f"Không tìm thấy file DEM tại: {dem_path}")
        return

    print("Đang nạp dữ liệu không gian cho huyện Bát Xát...")
    buildings = gpd.read_file(input_path)
    
    # 2. Mở file ảnh DEM
    with rasterio.open(dem_path) as dem:
        # 3. Đồng bộ hệ tọa độ (CRS)
        if buildings.crs != dem.crs:
            print(f"Đang chuyển hệ tọa độ về khớp với DEM ({dem.crs})...")
            buildings = buildings.to_crs(dem.crs)
            
        # 4. Lấy tọa độ tâm nhà (Centroid)
        print("Đang tính toán vị trí trung tâm của các mái nhà...")
        coords = [(geom.centroid.x, geom.centroid.y) for geom in buildings.geometry]
        
        # 5. Trích xuất độ cao (Sample)
        print(f"Đang trích xuất cao trình cho {len(buildings)} công trình...")
        
        elevations = []
        for i, val in enumerate(dem.sample(coords)):
            elevations.append(val[0])
            if i % 1000 == 0 and i > 0:
                print(f" > Đã xử lý {i} ngôi nhà...")
        
        buildings['elevation_z'] = elevations
    
    # 6. Lưu file kết quả
    buildings.to_file(output_path, driver='GeoJSON')
    print(f"Hoàn thành! Kết quả đã được lưu tại: {output_path}")

if __name__ == "__main__":
    overlay_dem()
