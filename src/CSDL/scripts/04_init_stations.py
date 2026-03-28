import os
import sys
from sqlalchemy import text

# Đảm bảo nhận diện được thư mục config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.db_config import get_engine

def sync_weather_stations():
    engine = get_engine()
    
    # Cấu hình từ script Bước 04 của bạn
    STATIONS = {
        "Y_TY": {"lat": 22.615564, "lon": 103.581009, "name": "Trạm Ý Tý"}, # Cao 2000m so với mực nước biển
        "SANG_MA_SAO": {"lat": 22.554365, "lon": 103.625492, "name": "Trạm Sàng Ma Sáo"}, # Cao 2300m so với mực nước biển
        "TRINH_TUONG": {"lat": 22.664940, "lon": 103.718175, "name": "Trạm Trịnh Tường"}, # Cao 300m -> 1000m so với mực nước biển
        "BAT_XAT": {"lat": 22.540857, "lon": 103.885786, "name": "Trạm Bát Xát"} # Bát Xát: 100 -> 500 so với mực nước biển
    }

    print("🛰️ Đang đồng bộ danh mục trạm khí tượng...")
    
    query = text("""
        INSERT INTO batxat_weather_stations (station_code, station_name, geom)
        VALUES (:code, :name, ST_SetSRID(ST_Point(:lon, :lat), 4326))
        ON CONFLICT (station_code) DO UPDATE 
        SET station_name = EXCLUDED.station_name,
            geom = EXCLUDED.geom;
    """)

    try:
        with engine.connect() as conn:
            for code, info in STATIONS.items():
                conn.execute(query, {
                    "code": code,
                    "name": info['name'],
                    "lon": info['lon'],
                    "lat": info['lat']
                })
            conn.commit()
        print("Đã khởi tạo 4 trạm khí tượng chủ lực!")
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    sync_weather_stations()