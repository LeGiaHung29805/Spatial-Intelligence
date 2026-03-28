import pandas as pd
import geopandas as gpd
import os
import sys
from shapely.geometry import Point
from sqlalchemy import text

# 1. Xử lý lỗi Import: Đảm bảo Python thấy thư mục 'config'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config.db_config import get_engine

def ingest_ml_samples():
    engine = get_engine()
    
    # 2. Đường dẫn file dữ liệu
    csv_path = "data/processed/landslide_training_data.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ Lỗi: Không tìm thấy file tại {csv_path}")
        return

    print(f"--- 🤖 ĐANG NẠP ĐIỂM MẪU HUẤN LUYỆN AI ---")
    df = pd.read_csv(csv_path)

    # 3. Chuyển đổi sang GeoDataFrame
    # Lưu ý: X là Kinh độ (Longitude), Y là Vĩ độ (Latitude)
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=[Point(x, y) for x, y in zip(df.Coord_X, df.Coord_Y)],
        crs="EPSG:4326"
    )

    # 4. Khớp tên cột với bảng batxat_ml_samples trong CSDL
    column_mapping = {
        'Elevation': 'elevation',
        'Slope': 'slope',
        'Dist_to_Water': 'dist_to_water',
        'Landslide_Label': 'landslide_label',
        'geometry': 'geom' # Cột hình học đổi tên thành 'geom'
    }
    
    # Đổi tên cột
    gdf = gdf.rename(columns=column_mapping)
    
    # SỬA LỖI TRỌNG TÂM: Khai báo lại cột tọa độ chủ đạo sau khi đổi tên
    gdf = gdf.set_geometry("geom")
    
    # Chỉ lấy các cột khớp với cấu trúc bảng CSDL
    final_cols = [col for col in column_mapping.values() if col in gdf.columns]
    gdf = gdf[final_cols]

    # 5. Đẩy vào CSDL PostGIS
    try:
        with engine.connect() as conn:
            print("🧹 Đang dọn dẹp dữ liệu mẫu cũ...")
            # SỬA LỖI SQL: Dùng text() và conn.commit() theo chuẩn SQLAlchemy 2.0
            conn.execute(text("TRUNCATE TABLE batxat_ml_samples RESTART IDENTITY CASCADE;"))
            conn.commit()
        
        print(f"📥 Đang đẩy {len(gdf)} mẫu vào bảng batxat_ml_samples...")
        gdf.to_postgis(
            name="batxat_ml_samples",
            con=engine,
            if_exists="append", # Dùng append vì đã TRUNCATE ở trên
            index=False
        )
        print("✅ THÀNH CÔNG: Các điểm mẫu đã sẵn sàng trong CSDL!")
        
    except Exception as e:
        print(f"❌ Lỗi hệ thống: {e}")

if __name__ == "__main__":
    ingest_ml_samples()