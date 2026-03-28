import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
from sqlalchemy import text
from keras.models import load_model
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

# Cấu hình đường dẫn hệ thống
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# 1. IMPORT CẤU HÌNH DB VÀ HÀM GỌI API (TỪ TẦNG CLIENT)
from src.CSDL.config.db_config import get_engine
from src.api_clients.module2_api.heatmap_client import trigger_heatmap_update

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

def load_latest_active_model(engine, target='FLOOD'):
    query = text("""
        SELECT model_path, scaler_path 
        FROM batxat_model_registry 
        WHERE model_target = :target AND is_active = TRUE 
        ORDER BY created_at DESC LIMIT 1
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {"target": target}).fetchone()
    if not result:
        raise FileNotFoundError(f"Không tìm thấy mô hình {target} nào đang Active!")
    return result.model_path, result.scaler_path

def update_spatial_landslide_risk(engine, rf_model, today_weather):
    logger.info("Đang tính toán Không gian (Spatial Prediction) cho toàn bộ khu vực...")
    
    query = text("SELECT id, elevation_z, dist_to_water FROM batxat_buildings")
    with engine.connect() as conn:
        buildings_df = pd.read_sql(query, conn)
    
    if buildings_df.empty:
        logger.warning("Không có dữ liệu công trình để dự báo không gian.")
        return 0.0

    buildings_df['Slope'] = 15.0 
    buildings_df['Elevation'] = buildings_df['elevation_z']
    buildings_df['Dist_to_Water'] = buildings_df['dist_to_water']
    buildings_df['precip_3d_BAT_XAT'] = today_weather['precip_3d_BAT_XAT']
    buildings_df['soil_BAT_XAT'] = today_weather.get('soil_BAT_XAT', 0.5)

    features = ['Slope', 'Elevation', 'Dist_to_Water', 'precip_3d_BAT_XAT', 'soil_BAT_XAT']
    X_spatial = buildings_df[features].fillna(0)

    # AI quét hàng loạt (Vectorized Prediction)
    probabilities = rf_model.predict_proba(X_spatial)[:, 1]
    buildings_df['landslide_prob'] = probabilities

    logger.info(f"Đang cập nhật {len(buildings_df)} điểm lên CSDL cho Heatmap...")
    
    # Ép kiểu dữ liệu về Python thuần để sửa lỗi psycopg2 InvalidSchemaName
    update_data = [
        {
            'p_id': int(row['id']), 
            'p_prob': float(row['landslide_prob'])
        } 
        for _, row in buildings_df.iterrows()
    ]
    
    update_query = text("UPDATE batxat_buildings SET landslide_prob = :p_prob WHERE id = :p_id")
    with engine.begin() as conn:
        conn.execute(update_query, update_data)
        
    return round(float(buildings_df['landslide_prob'].max() * 100), 2)


def run_integrated_predictions():
    engine = get_engine()
    logger.info("KÍCH HOẠT AI KÉP DỰ BÁO...")

    # NẠP MÔ HÌNH VÀ DỮ LIỆU
    try:
        m_path, s_path = load_latest_active_model(engine, 'FLOOD')
        lstm_flood = load_model(m_path)
        scaler = joblib.load(s_path)
        rf_landslide = joblib.load(MODELS_DIR / "rf_dynamic_landslide_model.pkl")
    except Exception as e:
        logger.error(f"Lỗi nạp mô hình: {e}"); return

    df = pd.read_csv(DATA_DIR / "flood_full_history.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')
    today = df.iloc[-1]

    day_of_year = today['datetime'].timetuple().tm_yday
    df['sin_season'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df['cos_season'] = np.cos(2 * np.pi * day_of_year / 365.25)
    for col in ['precip_3d_Y_TY', 'interaction_risk_Y_TY']:
        df[f'{col}_lag1'] = df[col].shift(1).fillna(0)

    lstm_features = [
        'precip_BAT_XAT', 'precip_3d_BAT_XAT', 'precip_5d_BAT_XAT', 'interaction_risk_BAT_XAT',
        'precip_3d_Y_TY_lag1', 'precip_5d_Y_TY', 'interaction_risk_Y_TY_lag1', 'basin_precip',
        'sin_season', 'cos_season'
    ]
    
    # DỰ BÁO NGẬP LỤT (LSTM)
    TIME_STEPS = 5
    if len(df) >= TIME_STEPS:
        X_input = scaler.transform(df.tail(TIME_STEPS)[lstm_features])
        X_3d = X_input.reshape(1, TIME_STEPS, len(lstm_features))
        flood_score = round(float(lstm_flood.predict(X_3d, verbose=0)[0][0] * 100), 2)
    else:
        flood_score = 0.0

    # DỰ BÁO SẠT LỞ (Random Forest)
    landslide_score = update_spatial_landslide_risk(engine, rf_landslide, today)

    # KẾT HỢP VÀ GHI DATABASE
    combined_risk = max(flood_score, landslide_score)
    status = 'DANGER' if combined_risk > 75 else 'WARNING' if combined_risk > 45 else 'SAFE'
    logger.info(f"Kết quả: Ngập {flood_score}% | Sạt lở đỉnh điểm {landslide_score}% -> {status}")

    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO batxat_daily_risk (forecast_date, flood_risk_pct, landslide_risk_pct, alert_level)
                VALUES (CURRENT_DATE, :f, :l, :s)
                ON CONFLICT (forecast_date) DO UPDATE SET 
                flood_risk_pct = EXCLUDED.flood_risk_pct, 
                landslide_risk_pct = EXCLUDED.landslide_risk_pct,
                alert_level = EXCLUDED.alert_level;
            """), {"f": flood_score, "l": landslide_score, "s": status})

            conn.execute(text("UPDATE batxat_system_state SET active_flood_level = 80.0 + (:f / 10.0)"), {"f": flood_score})

    except Exception as e:
        logger.error(f"Lỗi ghi DB: {e}")

    # 2. GỌI API ĐÁNH THỨC SPRING BOOT
    trigger_heatmap_update()
    
    logger.info("CHU KỲ HOÀN TẤT.")

if __name__ == "__main__":
    run_integrated_predictions()