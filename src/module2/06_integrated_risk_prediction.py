import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from keras.models import load_model
import os

# --- HÀM TÍNH TOÁN SẠT LỞ KHOA HỌC (TÍCH HỢP TỪ Ý TƯỞNG CỦA BẠN) ---
def calculate_scientific_landslide_risk(precip, water_proxy, slope, landcover):
    """
    Tính toán rủi ro sạt lở dựa trên ngưỡng tới hạn của khu vực Tây Bắc.
    Sử dụng dữ liệu từ Module 1 (Landcover) và Module 2 (Slope).
    """
    # 1. Xác định ngưỡng mưa dựa trên Thảm phủ (Module 1)
    # Đất trống/nông nghiệp (nhãn 1) nhạy cảm hơn rừng tự nhiên
    if landcover == 1:
        rain_threshold = 50.0       # mm/ngày (Mưa lớn tức thời)
        cum_rain_threshold = 80.0    # mm/3 ngày (Mưa tích lũy làm mềm đất)
    else:
        # Giả định các nhãn khác là rừng hoặc cây bụi có độ che phủ tốt hơn
        rain_threshold = 100.0 
        cum_rain_threshold = 150.0

    # 2. Tính toán trọng số rủi ro thành phần
    # Độ dốc: Ngưỡng tới hạn sạt lở vùng núi Việt Nam thường là 28 độ
    slope_risk = 1.0 if slope > 28 else (slope / 28.0)
    
    # Rủi ro do mưa (tức thời và tích lũy từ Bước 04)
    rain_risk = precip / rain_threshold
    cum_rain_risk = water_proxy / cum_rain_threshold

    # 3. Kết hợp có trọng số
    # Ưu tiên Độ dốc (40%) và sự tích tụ nước trong đất (30% mưa 3 ngày)
    total_ls_risk = (0.4 * slope_risk + 0.3 * rain_risk + 0.3 * cum_rain_risk) * 100
    
    return min(total_ls_risk, 100.0)

# --- HÀM ĐÁNH GIÁ TỔNG HỢP ---
def integrated_risk_assessment():
    print("BƯỚC 06: ĐANG CHẠY DỰ BÁO TỔNG HỢP (NGẬP + SẠT LỞ)...")
    MODELS_DIR = os.path.join("models")
    DATA_DIR = os.path.join("data","processed")
    # 1. CẤU HÌNH ĐƯỜNG DẪN
    MODEL_PATH = os.path.join(MODELS_DIR, "lstm_flood_seasonal.h5")
    SCALER_PATH = os.path.join(MODELS_DIR, "flood_scaler.pkl")
    DATA_PATH = os.path.join(DATA_DIR,"flood_training_data_lstm.csv")

    # 2. NẠP MÔ HÌNH VÀ DỮ LIỆU
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    df = pd.read_csv(DATA_PATH)
    
    # Lấy 7 ngày cuối để làm đầu vào cho LSTM
    recent_data = df.tail(7).copy()
    
    # 3. TẠO CÁC ĐẶC TRƯNG THỜI GIAN (MÙA)
    recent_data['datetime'] = pd.to_datetime(recent_data['datetime'])
    recent_data['month'] = recent_data['datetime'].dt.month
    recent_data['month_sin'] = np.sin(2 * np.pi * recent_data['month'] / 12)
    recent_data['month_cos'] = np.cos(2 * np.pi * recent_data['month'] / 12)

    # 4. DỰ BÁO NGẬP LỤT (Dùng mô hình LSTM đã train)
    features_list = ['precip', 'water_proxy', 'elevation', 'slope', 'landcover', 'month_sin', 'month_cos']
    scaled_data = scaler.transform(recent_data[features_list])
    input_seq = np.expand_dims(scaled_data, axis=0)
    
    flood_prob = model.predict(input_seq, verbose=0)
    flood_risk_score = float(flood_prob[0][0]) * 100

    # 5. DỰ BÁO SẠT LỞ (Dùng hàm khoa học tích hợp)
    # Lấy dữ liệu của ngày hiện tại (dòng cuối cùng)
    today = recent_data.iloc[-1]
    landslide_risk_score = calculate_scientific_landslide_risk(
        precip = today['precip'],
        water_proxy = today['water_proxy'],
        slope = today['slope'],
        landcover = today['landcover']
    )

    # 6. XUẤT BÁO CÁO VÀ LƯU FILE (QUAN TRỌNG CHO MODULE 3)
    combined_risk = max(flood_risk_score, landslide_risk_score)
    
    # Tạo DataFrame để lưu trữ
    report_df = pd.DataFrame([{
        'datetime': today['datetime'].strftime('%Y-%m-%d'),
        'flood_risk': round(flood_risk_score, 2),
        'landslide_risk': round(landslide_risk_score, 2),
        'combined_risk': round(combined_risk, 2),
        'precip': today['precip'],
        'status': 'DANGER' if combined_risk > 70 else 'WARNING' if combined_risk > 40 else 'SAFE'
    }])

    # Đường dẫn lưu file
    OUTPUT_PATH = r"e:\NCKH\GuardBaXat\Spatial-Intelligence\data\processed\daily_risk_report.csv"
    report_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "="*45)
    print(f"KẾT QUẢ PHÂN TÍCH RỦI RO HUYỆN BÁT XÁT")
    print(f"Ngày dự báo: {today['datetime'].strftime('%d/%m/%Y')}")
    print(f"Rủi ro Ngập lụt (LSTM):  {flood_risk_score:.2f}%")
    print(f"Rủi ro Sạt lở (Địa chất): {landslide_risk_score:.2f}%")
    print(f"ĐÃ LƯU BÁO CÁO TẠI: {OUTPUT_PATH}")
    print("="*45)

    return combined_risk

if __name__ == "__main__":
    integrated_risk_assessment()