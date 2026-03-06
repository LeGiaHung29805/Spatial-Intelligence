import geopandas as gpd
import os

def clean_data():
    # 1. Khai báo đường dẫn
    raw_gee_path = "data/shapefiles/infrastructure/BatXat_Buildings_GEE_Fixed.shp"
    output_dir = "data/processed"
    output_path = os.path.join(output_dir, "buildings_cleaned.geojson")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Đã tạo thư mục: {output_dir}")

    print("🚀 Đang bắt đầu làm sạch dữ liệu...")

    if not os.path.exists(raw_gee_path):
        print(f"❌ Lỗi: Không tìm thấy file tại {raw_gee_path}")
        return

    # 2. Đọc file
    gdf = gpd.read_file(raw_gee_path)
    print(f"🔍 Cột gốc từ GEE: {gdf.columns.tolist()}")

    # 3. Xử lý đổi tên cột bị cắt ngắn (Mapping)
    # Chúng ta tạo một "từ điển" để đổi 'area_in_me' về lại 'area_in_meters'
    name_map = {
        'area_in_me': 'area_in_meters',
        'AREA_IN_ME': 'area_in_meters' # Đề phòng trường hợp GEE viết hoa
    }
    
    # Chỉ đổi tên nếu cột đó thực sự tồn tại trong file
    gdf = gdf.rename(columns=name_map)
    
    # 4. Lọc cột với tên đã chuẩn hóa
    # Bây giờ bạn đã có thể gọi đúng tên 'area_in_meters'
    try:
        gdf = gdf[['geometry', 'area_in_meters', 'confidence']]
    except KeyError as e:
        print(f"❌ Vẫn thiếu cột: {e}. Vui lòng kiểm tra lại danh sách cột ở trên.")
        return

    # 5. Xuất file GeoJSON
    gdf.to_file(output_path, driver='GeoJSON')
    print(f"✅ Đã làm sạch dữ liệu thành công!")
    print(f"📍 File lưu tại: {output_path}")

if __name__ == "__main__":
    clean_data()
