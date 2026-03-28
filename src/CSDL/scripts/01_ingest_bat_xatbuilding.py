import geopandas as gpd
import os
import sys
from sqlalchemy import text
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.db_config import get_engine

def ingest_to_existing_table():
    engine = get_engine()
    input_path = "data/processed/buildings_with_hydro.geojson"
    
    print(f"--- 🏗️ ĐANG NẠP DỮ LIỆU VÀO CẤU TRÚC BẢNG CÓ SẴN ---")
    
    if not os.path.exists(input_path):
        print(f"❌ Không tìm thấy file: {input_path}")
        return

    # 1. Đọc dữ liệu
    gdf = gpd.read_file(input_path)

    # 2. Chuẩn bị cột để khớp với SQL
    # Đảm bảo các cột trong GeoDataFrame trùng tên hoàn toàn với bảng batxat_buildings
    # GEE/Preprocessing có thể tạo ra tên cột hơi khác, ta cần rename lại cho chuẩn
    column_mapping = {
        'geometry': 'geom',
        'elevation_z': 'elevation_z',
        'dist_to_water': 'dist_to_water',
        'area_in_meters': 'area_in_meters',
        'confidence': 'confidence'
    }
    
    # Lọc và đổi tên các cột cần thiết
    gdf = gdf.rename(columns=column_mapping)
    gdf = gdf.set_geometry("geom")
    # Chỉ giữ lại các cột có trong bảng SQL (trừ id vì nó tự tăng)
    final_columns = [col for col in column_mapping.values() if col in gdf.columns]
    gdf = gdf[final_columns]

    # 3. Làm sạch dữ liệu cũ (Tùy chọn)
    # Vì bạn nạp bản hoàn chỉnh, nên xóa dữ liệu nháp của các bước trước trong bảng
    with engine.connect() as conn:
        print("🧹 Đang dọn dẹp dữ liệu cũ trong bảng...")
        conn.execute(text("TRUNCATE TABLE batxat_buildings RESTART IDENTITY CASCADE;"))
        conn.commit()

    # 4. Nạp dữ liệu
    print(f"📥 Đang đẩy {len(gdf)} ngôi nhà vào PostgreSQL...")
    try:
        gdf.to_postgis(
            name="batxat_buildings", 
            con=engine, 
            if_exists="append", # QUAN TRỌNG: Dùng append để giữ nguyên cấu trúc bảng & Index
            index=False
        )
        print("✅ HOÀN THÀNH: Dữ liệu đã nằm gọn trong cấu trúc bảng của bạn!")
        
    except Exception as e:
        print(f"❌ Lỗi nạp dữ liệu: {e}")

if __name__ == "__main__":
    ingest_to_existing_table()