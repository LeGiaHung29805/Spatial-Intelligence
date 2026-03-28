import pandas as pd
import numpy as np
import os
import sys
from sqlalchemy import text

current_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.abspath(os.path.join(current_dir, '..')) 

if parent_dir not in sys.path:
    sys.path.append(parent_dir)
try:
    from config.db_config import get_engine
    print("✅ Đã kết nối thành công với Module cấu hình CSDL!")
except ModuleNotFoundError:
    print(f"❌ Vẫn không tìm thấy 'config'. Python đang tìm tại: {parent_dir}")
    sys.exit(1)

def ingest_weather_history_to_db(df_all):
    engine = get_engine()
    stations = ['Y_TY', 'SANG_MA_SAO', 'TRINH_TUONG', 'BAT_XAT']
    
    print("🔄 Đang chuyển đổi dữ liệu từ Wide sang Long...")
    
    all_rows = []
    
    for st in stations:
        # Cấu trúc map cột từ CSV -> CSDL
        cols_map = {
            'datetime': 'record_date',
            f'precip_{st}': 'precip',
            f'soil_{st}': 'soil_moist',
            f'precip_3d_{st}': 'precip_3d',
            f'precip_5d_{st}': 'precip_5d',
            f'soil_fatigue_{st}': 'soil_fatigue',
            f'interaction_risk_{st}': 'interaction_risk',
            'day_of_year': 'day_of_year',
            'sin_season': 'sin_season',
            'cos_season': 'cos_season'
        }
        
        # Kiểm tra xem các cột này có thực sự tồn tại trong file CSV không
        existing_cols = [c for c in cols_map.keys() if c in df_all.columns]
        
        if len(existing_cols) < 2: # Nếu chỉ có cột datetime thì bỏ qua trạm này
            print(f"⚠️ Cảnh báo: Trạm {st} không có dữ liệu trong file CSV.")
            continue
            
        df_st = df_all[existing_cols].copy()
        df_st = df_st.rename(columns=cols_map)
        df_st['station_code'] = st
        
        all_rows.append(df_st)

    if not all_rows:
        print("❌ Không có dữ liệu để nạp!")
        return

    df_long = pd.concat(all_rows, ignore_index=True)
    df_long['record_date'] = pd.to_datetime(df_long['record_date']).dt.date

    # Nạp vào CSDL
    try:
        with engine.begin() as conn:
            print(f"📥 Đang đẩy {len(df_long)} bản ghi vào batxat_weather_history...")
            # CASCADE giúp xóa sạch dữ liệu liên quan nếu có khóa ngoại
            conn.execute(text("TRUNCATE TABLE batxat_weather_history RESTART IDENTITY CASCADE;"))
            
            df_long.to_sql(
                'batxat_weather_history', 
                con=conn, 
                if_exists='append', 
                index=False,
                method='multi', # Tăng tốc độ insert
                chunksize=1000
            )
        print("✅ HOÀN TẤT: Toàn bộ dữ liệu khí tượng đã được đồng bộ.")
    except Exception as e:
        print(f"❌ Lỗi nạp dữ liệu: {e}")

if __name__ == "__main__":
    csv_path = "data/processed/flood_full_history.csv"
    if os.path.exists(csv_path):
        df_all = pd.read_csv(csv_path)
        ingest_weather_history_to_db(df_all)
    else:
        print(f"❌ Không tìm thấy file: {csv_path}")