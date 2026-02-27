import geopandas as gpd # Thư viện để xử lý dữ liệu bản đồ dạng Vector (mái nhà)

def clean_data():
    # 1. Khai báo đường dẫn file thô từ GEE và file kết quả
    raw_gee_path = "data/shapefiles/infrastructure/BatXat_Buildings_GEE_Fixed.shp"
    output_path = "data/processed/buildings_cleaned.geojson"
    
    # 2. Đọc file Shapefile vào bộ nhớ máy tính dưới dạng bảng (DataFrame)
    gdf = gpd.read_file(raw_gee_path)
    
    # 3. LỌC CỘT: Đây là dòng quan trọng nhất. 
    # SHP từ GEE có rất nhiều cột hệ thống. Chúng ta chỉ giữ lại:
    # - 'geometry': Hình dáng mái nhà (bắt buộc).
    # - 'area_in_meters': Diện tích mái nhà (để tính quy mô dân cư).
    # - 'confidence': Độ tin cậy của AI Google (để lọc bỏ nhà "ảo").
    gdf = gdf[['geometry', 'area_in_meters', 'confidence']]
    
    # 4. Xuất ra file GeoJSON. 
    # Tại sao dùng GeoJSON? Vì nó chỉ có 1 file duy nhất, dễ quản lý hơn bộ 4-5 file của SHP.
    gdf.to_file(output_path, driver='GeoJSON')
    print(f"✅ Đã làm sạch dữ liệu.")