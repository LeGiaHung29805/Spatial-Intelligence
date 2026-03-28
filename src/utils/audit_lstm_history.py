import pandas as pd
import numpy as np
import joblib
import os
import warnings

# Tắt các cảnh báo hệ thống không cần thiết
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model

def audit_historical_storms():
    print("🕵️ ĐANG KIỂM CHỨNG LỊCH SỬ DỰ BÁO CỦA LSTM (ĐÃ TÍCH HỢP SEASONALITY)...")
    
    MODELS_DIR = "models"
    DATA_DIR = "data/processed"
    
    # 1. NẠP MÔ HÌNH VÀ THƯỚC ĐO
    try:
        lstm_flood = load_model(os.path.join(MODELS_DIR, "lstm_flood_model.keras"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler_lstm.pkl"))
    except Exception as e:
        print(f"❌ Lỗi nạp mô hình: {e}")
        return

    # 2. NẠP DỮ LIỆU 10 NĂM
    df = pd.read_csv(os.path.join(DATA_DIR, "flood_full_history.csv"))
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    for col in ['precip_3d_Y_TY', 'interaction_risk_Y_TY']:
        df[f'{col}_lag1'] = df[col].shift(1).fillna(0)

    # ĐÃ THÊM: Tính năng chu kỳ Mùa vụ (Sin/Cos Seasonality)
    lstm_features = [
        'precip_BAT_XAT', 'precip_3d_BAT_XAT', 'precip_5d_BAT_XAT', 'interaction_risk_BAT_XAT',
        'precip_3d_Y_TY_lag1', 'precip_5d_Y_TY', 'interaction_risk_Y_TY_lag1', 'basin_precip',
        'sin_season', 'cos_season'
    ]

    # 3. LỌC TÌM CÁC NGÀY KIỂM CHỨNG (TEST CASES)
    # Lấy 10 ngày mưa lớn nhất lịch sử (Top 10 Storms)
    top_storms = df.nlargest(10, 'basin_precip')
    
    # Lấy 5 ngày khô hạn tuyệt đối (Mưa = 0) để xem AI có bị hoang tưởng không
    dry_days = df[df['basin_precip'] == 0].sample(n=5, random_state=42)
    
    # Gộp lại và sắp xếp theo thời gian
    test_cases = pd.concat([top_storms, dry_days]).sort_values('datetime')
    # 3 trận siêu bão (Chắc chắn 99%)
    # top_storms = df.nlargest(3, 'basin_precip')
    
    # # 5 ngày MƯA VỪA, NGẬP LẤP XẤP (Vùng xám: Lượng mưa từ 25mm đến 40mm)
    # medium_rain_days = df[(df['basin_precip'] >= 25) & (df['basin_precip'] <= 40)].sample(n=5, random_state=42)
    
    # # 2 ngày nắng hạn tuyệt đối (Chắc chắn 0%)
    # dry_days = df[df['basin_precip'] == 0].sample(n=2, random_state=42)

    print("\n" + "="*75)
    print(f"{'NGÀY THÁNG':<12} | {'MƯA HÔM ĐÓ':<12} | {'TÍCH LŨY 3 NGÀY':<15} | {'DỰ BÁO LỤT (LSTM)':<18}")
    print("-" * 75)

    # 4. CHẠY KIỂM TOÁN TỪNG NGÀY
    for idx, row in test_cases.iterrows():
        end_date = row['datetime']
        
        # Lấy 5 ngày tính đến thời điểm đó (Cửa sổ trượt)
        df_window = df[df['datetime'] <= end_date].tail(5)
        if len(df_window) < 5: continue

        X_raw = df_window[lstm_features]
        X_scaled = scaler.transform(X_raw)
        X_3d = X_scaled.reshape(1, 5, len(lstm_features))

        prob = lstm_flood.predict(X_3d, verbose=0)[0][0] * 100

        date_str = end_date.strftime('%Y-%m-%d')
        rain = f"{row['basin_precip']:.1f} mm"
        rain_3d = f"{row['precip_3d_BAT_XAT']:.1f} mm"
        pred_str = f"{prob:.2f} %"

        # Đánh dấu bằng Icon để dễ nhìn
        if row['basin_precip'] >= 40:
            marker = "🔴 BÃO LỚN"
        elif row['basin_precip'] == 0:
            marker = "☀️ NẮNG HẠN"
        else:
            marker = "🌧️ MƯA VỪA"

        print(f"{date_str:<12} | {rain:<12} | {rain_3d:<15} | {pred_str:<8} ({marker})")
        
    print("="*75)

if __name__ == "__main__":
    audit_historical_storms()