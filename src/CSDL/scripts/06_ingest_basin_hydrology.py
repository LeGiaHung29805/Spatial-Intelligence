import pandas as pd
import os
import sys
from sqlalchemy import text

# --- PHẦN 1: SỬA LỖI ĐƯỜNG DẪN (PATH) ---
current_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.abspath(os.path.join(current_dir, '..')) 

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from config.db_config import get_engine
    print("✅ Kết nối Module cấu hình thành công!")
except ModuleNotFoundError:
    print(f"❌ Không tìm thấy 'config'. Python đang tìm tại: {parent_dir}")
    sys.exit(1)

def ingest_basin_hydrology():
    engine = get_engine()
    csv_path = "data/processed/flood_full_history.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ Không tìm thấy file: {csv_path}")
        return

    print("📊 Đang trích xuất chỉ số thủy văn lưu vực...")
    df = pd.read_csv(csv_path)

    # 1. Map tên cột từ CSV (Bước 04) sang CSDL
    # CSV: datetime, basin_precip, water_level, is_event, impact_score
    # SQL: record_date, basin_precip, estimated_water_level, is_event, real_impact_score
    cols_mapping = {
        'datetime': 'record_date',
        'basin_precip': 'basin_precip',
        'water_level': 'estimated_water_level',
        'is_event': 'is_event',
        'impact_score': 'real_impact_score'
    }

    # Lọc lấy các cột cần thiết
    df_basin = df[list(cols_mapping.keys())].copy()
    df_basin = df_basin.rename(columns=cols_mapping)

    # 2. Chuẩn hóa dữ liệu
    df_basin['record_date'] = pd.to_datetime(df_basin['record_date']).dt.date
    df_basin['is_event'] = df_basin['is_event'].astype(bool)

    # 3. Nạp vào CSDL
    try:
        with engine.begin() as conn:
            print(f"📥 Đang nạp {len(df_basin)} ngày dữ liệu thủy văn tổng hợp...")
            # Làm sạch dữ liệu cũ
            conn.execute(text("TRUNCATE TABLE batxat_basin_hydrology CASCADE;"))
            
            df_basin.to_sql(
                'batxat_basin_hydrology', 
                con=conn, 
                if_exists='append', 
                index=False,
                method='multi'
            )
        print("✅ THÀNH CÔNG: Bảng batxat_basin_hydrology đã sẵn sàng!")
    except Exception as e:
        print(f"❌ Lỗi nạp CSDL: {e}")

if __name__ == "__main__":
    ingest_basin_hydrology()