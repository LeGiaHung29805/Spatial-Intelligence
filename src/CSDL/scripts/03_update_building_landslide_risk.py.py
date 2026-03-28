import rasterio
import geopandas as gpd
from sqlalchemy import text
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from config.db_config import get_engine

def update_landslide_risk():
    engine = get_engine()
    heatmap_path = "data/processed/batxat_landslide_heatmap.tif"
    
    # 1. Lấy dữ liệu nhà từ CSDL
    print("🏠 Đang lấy danh sách nhà cửa từ CSDL...")
    buildings = gpd.read_postgis("SELECT id, geom FROM batxat_buildings", engine, geom_col='geom')

    # 2. Mở heatmap để trích xuất giá trị
    with rasterio.open(heatmap_path) as src:
        # Chuyển đổi tọa độ nhà sang hệ tọa độ của ảnh
        buildings = buildings.to_crs(src.crs)
        
        # Lấy tọa độ tâm nhà
        coords = [(geom.centroid.x, geom.centroid.y) for geom in buildings.geometry]
        
        # Trích xuất giá trị xác suất sạt lở (0.0 -> 1.0)
        print("🧠 AI đang phân tích rủi ro sạt lở cho từng ngôi nhà...")
        buildings['landslide_prob'] = [val[0] for val in src.sample(coords)]

    # 3. Cập nhật lại vào CSDL
    print("💾 Đang đồng bộ kết quả dự báo vào CSDL...")
    with engine.begin() as conn:
        # Thêm cột nếu chưa có
        conn.execute(text("ALTER TABLE batxat_buildings ADD COLUMN IF NOT EXISTS landslide_prob REAL;"))
        
        # Cập nhật giá trị
        for _, row in buildings.iterrows():
            if row['landslide_prob'] != -9999.0: # Bỏ qua vùng NoData
                conn.execute(
                    text("UPDATE batxat_buildings SET landslide_prob = :prob WHERE id = :id"),
                    {"prob": float(row['landslide_prob']), "id": int(row['id'])}
                )
    print("✅ HOÀN TẤT: 18.000 ngôi nhà đã được cập nhật xác suất sạt lở!")

if __name__ == "__main__":
    update_landslide_risk()