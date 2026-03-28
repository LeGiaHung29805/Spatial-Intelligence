import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine, text
import os

# Import cấu hình từ file config.py cùng thư mục
from config import SQLALCHEMY_DATABASE_URL

def sync_normalized_db():
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    processed_dir = "data/processed"
    
    # CHỈ ĐỊNH RÕ 3 KỊCH BẢN CỦA BẠN (Không dùng *m tự động nữa)
    scenarios = [85, 90, 95]
    
    print(f"🎯 Bắt đầu tiến trình ETL cho 3 kịch bản: {scenarios}...\n")

    # 1. XÓA DỮ LIỆU CŨ ĐỂ CHẠY LẠI AN TOÀN
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE flood_impacts, batxat_buildings RESTART IDENTITY CASCADE;"))
        conn.commit()
        print("🧹 Đã làm sạch cơ sở dữ liệu cũ.")

    # 2. NẠP DỮ LIỆU TĨNH (Lấy file 85m làm gốc để trích xuất tọa độ nhà)
    base_level = scenarios[0]
    base_file = os.path.join(processed_dir, f"final_analysis_{base_level}m.geojson")
    
    print(f"🏗️ Đang trích xuất thông tin tòa nhà gốc từ file {base_level}m...")
    gdf_base = gpd.read_file(base_file)
    
    # Cấp ID cho từng nhà
    gdf_base['building_id'] = gdf_base.index + 1
    
    static_cols = ['building_id', 'geometry', 'area_in_meters', 'elevation_z', 'building_type']
    gdf_static = gdf_base[static_cols].copy()
    
    # 1. Đổi tên cột
    gdf_static = gdf_static.rename(columns={'geometry': 'geom'})
    
    # 2. KHẲNG ĐỊNH VỚI GEOPANDAS: "Cột 'geom' chính là cột không gian!"
    gdf_static = gdf_static.set_geometry('geom')
    
    # 3. Bây giờ thì đẩy lên PostGIS thoải mái
    gdf_static.to_postgis('batxat_buildings', engine, if_exists='append', index=False)
    print(f"   ✅ Đã nạp {len(gdf_static)} công trình vào bảng batxat_buildings.\n")

    # 3. NẠP DỮ LIỆU ĐỘNG CHO CHÍNH XÁC 3 KỊCH BẢN
    for level in scenarios:
        file_path = os.path.join(processed_dir, f"final_analysis_{level}m.geojson")
        print(f"🌊 Đang nạp kịch bản lũ: {level}m...")
        
        gdf_scenario = gpd.read_file(file_path)
        
        # Cấp lại ID y hệt như trên để liên kết khóa ngoại
        gdf_scenario['building_id'] = gdf_scenario.index + 1
        gdf_scenario['flood_level'] = level
        
        # Phòng hờ nếu file của bạn chưa có cột flood_depth
        if 'flood_depth' not in gdf_scenario.columns:
            gdf_scenario['flood_depth'] = gdf_scenario['elevation_z'].apply(lambda z: max(0, level - z))
        
        # Chỉ lấy các cột thay đổi (không lấy geometry)
        dynamic_cols = ['building_id', 'flood_level', 'flood_depth', 'risk_status', 'capacity']
        df_dynamic = pd.DataFrame(gdf_scenario[dynamic_cols])
        
        df_dynamic.to_sql('flood_impacts', engine, if_exists='append', index=False)
        print(f"   ✅ Xong kịch bản {level}m!")

    print("\n✨ THÀNH CÔNG! Dữ liệu đã vào CSDL chuẩn.")

if __name__ == "__main__":
    sync_normalized_db()