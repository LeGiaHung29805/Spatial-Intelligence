import pandas as pd
import numpy as np
import os
import sys
import uuid
import logging
from datetime import datetime, timedelta
from sqlalchemy import text
from pathlib import Path

# Cấu hình logging và đường dẫn
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Import cấu hình DB và các API Client từ Tầng Giao tiếp Mạng
from src.CSDL.config.db_config import get_engine
from src.api_clients.module2_api.weather_client import fetch_vc_history_range, fetch_soil_fatigue_api

STATIONS = {
    "Y_TY": {"lat": 22.615564, "lon": 103.581009},
    "SANG_MA_SAO": {"lat": 22.554365, "lon": 103.625492},
    "TRINH_TUONG": {"lat": 22.664940, "lon": 103.718175},
    "BAT_XAT": {"lat": 22.540857, "lon": 103.885786}
}

DATA_DIR = PROJECT_ROOT / "data" / "processed"
FULL_HISTORY_CSV = DATA_DIR / "flood_full_history.csv"
ACTIVE_CONTEXT_CSV = DATA_DIR / "flood_training_data_lstm.csv"
SPATIAL_DATA_CSV = DATA_DIR / "landslide_training_data.csv"
DYNAMIC_FUSION_CSV = DATA_DIR / "dynamic_landslide_fusion.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)

def run_manual_flood_simulation(target_level, engine):
    sim_id = str(uuid.uuid4())
    logger.info(f"Đang chạy giả lập: {target_level}m (ID: {sim_id})")
    query_sql = text("SELECT building_id, flood_depth, risk_status FROM simulate_flood_risk(:level)")
    try:
        with engine.begin() as conn:
            results = conn.execute(query_sql, {"level": target_level}).fetchall()
            insert_list = [
                {
                    "simulation_id": sim_id, "input_level": target_level,
                    "building_id": row.building_id, "depth_impact": row.flood_depth,
                    "risk_status": row.risk_status
                }
                for row in results if row.risk_status != 'An toàn'
            ]
            if insert_list:
                conn.execute(
                    text("""
                        INSERT INTO batxat_flood_simulation 
                        (simulation_id, input_level, building_id, depth_impact, risk_status)
                        VALUES (:simulation_id, :input_level, :building_id, :depth_impact, :risk_status)
                    """), insert_list
                )
                logger.info(f"Đã cập nhật {len(insert_list)} công trình vào kịch bản lũ.")
        return sim_id
    except Exception as e:
        logger.error(f"Lỗi chạy mô phỏng ngập lụt: {e}")
        return None

def get_dynamic_impact_score(current_water_index, engine):
    flood_level_meters = min(78.0 + (current_water_index / 10.0), 88.0)
    query = text("SELECT COUNT(*) FROM simulate_flood_risk(:level) WHERE risk_status LIKE 'Nguy cơ%';")
    try:
        with engine.connect() as conn:
            return conn.execute(query, {"level": flood_level_meters}).scalar()
    except Exception as e:
        return 0

def super_data_fusion_engine():
    engine = get_engine()
    now = datetime.now()
    today_str = now.strftime('%Y-%m-%d')
    logger.info("ĐANG ĐỒNG BỘ HỆ THỐNG ĐA NGUỒN...")
    
    try:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM batxat_flood_simulation WHERE created_at < NOW() - INTERVAL '24 hours'"))
    except Exception as e: pass

    df_all = pd.read_csv(FULL_HISTORY_CSV, parse_dates=['datetime']).set_index('datetime').sort_index() if FULL_HISTORY_CSV.exists() else pd.DataFrame()

    lookback_days = 15
    start_check_dt = (now - timedelta(days=lookback_days)).replace(hour=0, minute=0, second=0, microsecond=0)
    expected_range = pd.date_range(start=start_check_dt, end=now.date(), freq='D')
    
    precip_cols = [f'precip_{name}' for name in STATIONS.keys()]
    if not df_all.empty:
        valid_dates = df_all.index[df_all[precip_cols].notna().all(axis=1)]
        missing_dates = expected_range.difference(valid_dates)
    else:
        missing_dates = expected_range

    gap_start_dt = missing_dates.min() if not missing_dates.empty else now
    gap_start_str = gap_start_dt.strftime('%Y-%m-%d')

    if not missing_dates.empty:
        logger.info(f"Thiếu {len(missing_dates)} ngày. Đang gọi API vá lỗi...")
        for name, coords in STATIONS.items():
            rain_map = fetch_vc_history_range(coords['lat'], coords['lon'], gap_start_str, today_str)
            fatigue_map = fetch_soil_fatigue_api(coords['lat'], coords['lon'], lookback_days)
            
            for date_str, rain_val in rain_map.items():
                dt_obj = pd.to_datetime(date_str)
                df_all.loc[dt_obj, f'precip_{name}'] = rain_val
                if date_str in fatigue_map:
                    df_all.loc[dt_obj, f'soil_fatigue_{name}'] = fatigue_map[date_str]
    else:
        logger.info("Dữ liệu 15 ngày gần nhất đã đầy đủ.")

    df_all = df_all.replace(-999.0, np.nan).sort_index()
    for name in STATIONS.keys():
        df_all[f'precip_{name}'] = df_all[f'precip_{name}'].fillna(0)
        df_all[f'soil_{name}'] = df_all.get(f'soil_{name}', pd.Series(0.5, index=df_all.index)).ffill().fillna(0.5)
        df_all[f'soil_fatigue_{name}'] = df_all.get(f'soil_fatigue_{name}', pd.Series(0.0, index=df_all.index)).fillna(0.0)
        
        df_all[f'precip_3d_{name}'] = df_all[f'precip_{name}'].rolling(window=3, min_periods=1).sum().fillna(0)
        df_all[f'precip_5d_{name}'] = df_all[f'precip_{name}'].rolling(window=5, min_periods=1).sum().fillna(0)
        df_all[f'interaction_risk_{name}'] = df_all[f'precip_3d_{name}'] * df_all[f'soil_{name}']

    for col in ['elevation', 'slope', 'landcover', 'is_event']:
        if col in df_all.columns:
            df_all[col] = df_all[col].ffill().fillna(0)

    df_all['basin_precip'] = (df_all['precip_Y_TY']*0.4 + df_all['precip_SANG_MA_SAO']*0.2 + 
                               df_all['precip_TRINH_TUONG']*0.2 + df_all['precip_BAT_XAT']*0.2)
    df_all['day_of_year'] = df_all.index.dayofyear
    df_all['sin_season'] = np.sin(2 * np.pi * df_all['day_of_year'] / 365.25)
    df_all['cos_season'] = np.cos(2 * np.pi * df_all['day_of_year'] / 365.25)
    df_all = df_all.fillna(0)

    water_levels = [0]
    for p in df_all['basin_precip']:
        water_levels.append(p + (0.85 * water_levels[-1]))
    df_all['water_level'] = water_levels[1:]

    logger.info("Đang mô phỏng thiệt hại bằng AI (Direct DB Query)...")
    sync_mask = df_all.index >= gap_start_dt
    if sync_mask.any():
        df_all.loc[sync_mask, 'impact_score'] = [
            get_dynamic_impact_score(wl, engine) for wl in df_all.loc[sync_mask, 'water_level']
        ]
    df_all['impact_score'] = df_all['impact_score'].fillna(0)

    try:
        with engine.begin() as conn:
            df_sync = df_all[df_all.index >= gap_start_dt]
            for d_idx, row in df_sync.iterrows():
                curr_date = d_idx.date()
                for st in STATIONS.keys():
                    conn.execute(text("""
                        INSERT INTO batxat_weather_history (precip, precip_3d, sin_season, record_date, station_code)
                        VALUES (:p, :p3, :ss, :rd, :sc)
                        ON CONFLICT (record_date, station_code) DO UPDATE SET 
                        precip=EXCLUDED.precip, precip_3d=EXCLUDED.precip_3d, sin_season=EXCLUDED.sin_season;
                    """), {'p': float(row[f'precip_{st}']), 'p3': float(row[f'precip_3d_{st}']), 
                           'ss': float(row['sin_season']), 'rd': curr_date, 'sc': st})

                conn.execute(text("""
                    INSERT INTO batxat_basin_hydrology (basin_precip, estimated_water_level, real_impact_score, record_date)
                    VALUES (:bp, :wl, :is, :rd)
                    ON CONFLICT (record_date) DO UPDATE SET 
                    basin_precip=EXCLUDED.basin_precip, estimated_water_level=EXCLUDED.estimated_water_level, real_impact_score=EXCLUDED.real_impact_score;
                """), {'bp': float(row['basin_precip']), 'wl': float(row['water_level']), 
                       'is': float(row['impact_score']), 'rd': curr_date})
        logger.info(f"Đã đồng bộ CSDL thành công đến ngày {today_str}!")
    except Exception as e: logger.error(f"Lỗi nạp CSDL: {e}")

    df_all.reset_index().to_csv(FULL_HISTORY_CSV, index=False)
    df_all.tail(750).reset_index().to_csv(ACTIVE_CONTEXT_CSV, index=False)
    
    if SPATIAL_DATA_CSV.exists():
        df_spatial = pd.read_csv(SPATIAL_DATA_CSV)
        weather_lite = df_all[['basin_precip', 'precip_3d_BAT_XAT', 'soil_BAT_XAT']].tail(100).copy()
        weather_lite['key'], df_spatial['key'] = 1, 1
        pd.merge(df_spatial, weather_lite, on='key').drop('key', axis=1).to_csv(DYNAMIC_FUSION_CSV, index=False)
        
    logger.info("HOÀN TẤT MODULE DATA FUSION.")

if __name__ == "__main__":
    engine = get_engine()
    super_data_fusion_engine()
    run_manual_flood_simulation(83.5, engine)