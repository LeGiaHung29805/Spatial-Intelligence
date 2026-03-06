import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout
import joblib
import os

def train_flood_model_seasonal():
    print("🌊 ĐANG HUẤN LUYỆN LSTM: NHẬN DIỆN MÙA & ĐỊA HÌNH BÁT XÁT...")
    MODELS_DIR = os.path.join("models")
    # 1. Đọc dữ liệu
    file_path = os.path.join(MODELS_DIR, "flood_training_data_lstm.csv")
    if not os.path.exists(file_path):
        print("❌ Không tìm thấy file CSV. Hãy chạy file 04 trước!")
        return
        
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # --- TÍNH NĂNG THEO MÙA (Cần thiết để khớp với Scaler) ---
    df['month'] = df['datetime'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # 2. Gán nhãn (Labeling)
    threshold = df['water_proxy'].quantile(0.90)
    df['label'] = (df['water_proxy'] > threshold).astype(int)

    # 3. CHUẨN HÓA ĐỒNG NHẤT (Sửa lỗi ValueError ở Bước 06)
    # Danh sách features phải khớp 100% với file 06
    features = ['precip', 'water_proxy', 'elevation', 'slope', 'landcover', 'month_sin', 'month_cos']
    
    # Khởi tạo Scaler
    scaler = MinMaxScaler()
    
    # QUAN TRỌNG: fit_transform trên toàn bộ khối 7 cột để Scaler "nhớ" tên đặc trưng
    # Ngay cả khi elevation/slope không đổi, việc fit chung như thế này giúp tránh lỗi Feature Names
    df[features] = scaler.fit_transform(df[features])

    # 4. Tạo cửa sổ thời gian (7 ngày)
    X, y = [], []
    window = 7
    values = df[features].values # Lấy giá trị đã chuẩn hóa
    target = df['label'].values
    
    for i in range(len(values) - window):
        X.append(values[i : i + window])
        y.append(target[i + window])
        
    X, y = np.array(X), np.array(y)

    # 5. Xây dựng cấu trúc LSTM
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid') 
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 6. Huấn luyện
    print(f"🚀 Huấn luyện AI với đầu vào: {X.shape}")
    model.fit(X, y, epochs=30, batch_size=16, validation_split=0.1, verbose=1)

    # 7. Lưu kết quả
    # model_dir = r"e:\NCKH\GuardBaXat\Spatial-Intelligence\models"
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    
    model.save(os.path.join(MODELS_DIR, "lstm_flood_seasonal.h5"))
    
    # Lưu Scaler đã "học" đủ 7 cột
    joblib.dump(scaler, os.path.join(MODELS_DIR, "flood_scaler.pkl"))
    print("✅ Đã lưu Model và Scaler đồng bộ!")

if __name__ == "__main__":
    train_flood_model_seasonal()