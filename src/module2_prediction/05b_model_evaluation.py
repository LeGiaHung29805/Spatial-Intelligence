import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def evaluate_dual_rf_system():
    print("🧪 ĐANG VẼ ĐỒ THỊ DIỄN BIẾN RỦI RO NGẬP LỤT (DUAL-RF)...")
    MODELS_DIR = "models"
    DATA_DIR = "data/processed"
    
    # 1. NẠP DỮ LIỆU
    df = pd.read_csv(os.path.join(DATA_DIR, "flood_full_history.csv"))
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')

    # Tính lại Lag Features (Khớp với Bước 05)
    for col in ['precip_3d_Y_TY', 'interaction_risk_Y_TY']:
        df[f'{col}_lag1'] = df[col].shift(1).fillna(0)
    
    rf_flood_features = [
        'precip_BAT_XAT', 'precip_3d_BAT_XAT', 'precip_5d_BAT_XAT', 'interaction_risk_BAT_XAT',
        'precip_3d_Y_TY_lag1', 'precip_5d_Y_TY', 'interaction_risk_Y_TY_lag1', 'basin_precip'
    ]
    
    # Loại bỏ NaN do phép tính lag
    df = df.dropna(subset=rf_flood_features)

    # 2. NẠP MÔ HÌNH RANDOM FOREST NGẬP LỤT
    try:
        rf_flood = joblib.load(os.path.join(MODELS_DIR, "rf_flood_model.pkl"))
        print("✅ Đã kết nối với bộ não AI Ngập Lụt (Random Forest).")
    except Exception as e:
        print(f"❌ Lỗi nạp mô hình: {e}")
        return

    # 3. DỰ BÁO XÁC SUẤT
    X_flood = df[rf_flood_features]
    y_pred_prob = rf_flood.predict_proba(X_flood)[:, 1] # Lấy xác suất của nhãn 1 (Lũ)
    
    y_true_flood = ((df['water_level'] > df['water_level'].quantile(0.92)) | (df['is_event'] == 1)).astype(int)
    y_pred_final = rf_flood.predict(X_flood)

    # 4. TRỰC QUAN HÓA: VẼ BIỂU ĐỒ BÃO YAGI
    mask_recent = df['datetime'] >= '2023-01-01' # Chỉ vẽ từ 2023 cho dễ nhìn
    
    plt.figure(figsize=(15, 6))
    plt.plot(df['datetime'][mask_recent], y_pred_prob[mask_recent], label='Xác suất Ngập lụt (AI)', color='teal')
    plt.axhline(y=0.5, color='red', linestyle='--', label='Ngưỡng Báo động (50%)')
    
    # Đánh dấu Bão Yagi (T9/2024)
    yagi_mask = (df['datetime'] >= '2024-09-07') & (df['datetime'] <= '2024-09-12')
    plt.fill_between(df['datetime'][yagi_mask], 0, 1, color='red', alpha=0.3, label='Siêu bão Yagi (Thực tế)')

    plt.title('Đồ thị Cảnh báo Sớm Ngập Lụt - GuardBaXat AI (Giai đoạn 2023 - 2024)')
    plt.ylabel('Mức độ rủi ro (0 - 1.0)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show() # TẮT CỬA SỔ NÀY ĐỂ XEM MA TRẬN NHẦM LẪN

    # 5. MA TRẬN NHẦM LẪN (Confusion Matrix)
    cm = confusion_matrix(y_true_flood, y_pred_final)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['An toàn', 'Ngập Lụt'], yticklabels=['An toàn', 'Ngập Lụt'])
    plt.title('Ma trận nhầm lẫn (Confusion Matrix) - Ngập Lụt')
    plt.xlabel('AI Dự báo')
    plt.ylabel('Thực tế')
    plt.show()

    print("\n✅ HOÀN TẤT THẨM ĐỊNH! Module 2 đã chính thức khép lại.")

if __name__ == "__main__":
    evaluate_dual_rf_system()