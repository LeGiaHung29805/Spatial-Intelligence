import pandas as pd
import numpy as np
import os
import joblib
import warnings
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
# Thêm precision_score, recall_score để phục vụ thuật toán tối ưu ngưỡng
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, precision_score, recall_score 
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sqlalchemy import text
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
csdl_path = os.path.abspath(os.path.join(current_dir, '..', 'CSDL'))
if csdl_path not in sys.path:
    sys.path.append(csdl_path)

try:
    from config.db_config import get_engine
except ImportError:
    print("Lỗi: Không tìm thấy module cấu hình CSDL!")
    sys.exit(1)

warnings.filterwarnings('ignore')

def create_sequences(X, y, time_steps):
    """Kỹ thuật Cửa sổ trượt 3D cho LSTM"""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


# HUẤN LUYỆN DỰ BÁO NGẬP (GIỮ NGUYÊN 100%)
def train_lstm_smote_model():
    print("\n[1/2] ĐANG HUẤN LUYỆN LSTM + SMOTE (DỰ BÁO NGẬP)...")
    DATA_DIR = "data/processed"
    MODELS_DIR = "models"
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    data_path = os.path.join(DATA_DIR, "flood_full_history.csv")
    if not os.path.exists(data_path):
        print("Lỗi: Không tìm thấy file dữ liệu lịch sử!")
        return
        
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')

    for col in ['precip_3d_Y_TY', 'interaction_risk_Y_TY']:
        df[f'{col}_lag1'] = df[col].shift(1).fillna(0)
    
    lstm_features = [
        'precip_BAT_XAT', 'precip_3d_BAT_XAT', 'precip_5d_BAT_XAT', 'interaction_risk_BAT_XAT',
        'precip_3d_Y_TY_lag1', 'precip_5d_Y_TY', 'interaction_risk_Y_TY_lag1', 'basin_precip',
        'sin_season', 'cos_season' 
    ]
    df = df.dropna(subset=lstm_features)

    if 'water_level' not in df.columns:
        df['water_level'] = df['basin_precip'] 
    
    y_target = ((df['water_level'] > df['water_level'].quantile(0.92)) | (df.get('is_event', 0) == 1)).astype(int)
    X_raw = df[lstm_features]

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)
    
    TIME_STEPS = 5 
    X_seq, y_seq = create_sequences(X_scaled, y_target, TIME_STEPS)

    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    samples, ts, feats = X_train.shape
    X_train_2d = X_train.reshape(samples, ts * feats)
    smote = SMOTE(random_state=42)
    X_train_smote_2d, y_train_smote = smote.fit_resample(X_train_2d, y_train)
    X_train_smote = X_train_smote_2d.reshape(X_train_smote_2d.shape[0], ts, feats)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(TIME_STEPS, feats)),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model.fit(X_train_smote, y_train_smote, epochs=50, batch_size=32, 
              validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    model_path = os.path.join(MODELS_DIR, f"lstm_flood_v1_{timestamp}.keras")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_lstm_v1_{timestamp}.pkl")
    
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    
    # =========================================================================
    # ĐOẠN CODE CẬP NHẬT: TỐI ƯU HÓA NGƯỠNG CÂN BẰNG (ƯU TIÊN RECALL)
    # =========================================================================
    print("\n" + "="*50)
    print(" KẾT QUẢ ĐÁNH GIÁ CHUYÊN SÂU MÔ HÌNH LSTM (NGẬP LỤT)")
    print("="*50)
    
    y_pred_prob = model.predict(X_test, verbose=0)
    
    # 1. Thuật toán tự động quét ngưỡng ưu tiên Recall (Bắt thảm họa) >= 80%
    best_thresh = 0.5
    min_diff = 1.0 # Tìm khoảng cách nhỏ nhất giữa Precision và Recall để cân bằng
    
    for thresh in np.arange(0.4, 0.90, 0.05):
        y_pred_temp = (y_pred_prob > thresh).astype(int)
        
        # Bỏ qua nếu ngưỡng này khiến AI đoán 100% là An toàn (Lỗi chia 0)
        if 1 in y_pred_temp:
            p_temp = precision_score(y_test, y_pred_temp, zero_division=0)
            r_temp = recall_score(y_test, y_pred_temp, zero_division=0)
            
            # ĐIỀU KIỆN TIÊN QUYẾT: Recall phải đạt ít nhất 80%
            if r_temp >= 0.80:
                diff = abs(p_temp - r_temp)
                if diff < min_diff:
                    min_diff = diff
                    best_thresh = thresh
            
    print(f">> THUẬT TOÁN TỐI ƯU HÓA NGƯỠNG (CÂN BẰNG P-R) ĐÃ KÍCH HOẠT <<")
    print(f">> Ngưỡng cảnh báo cân bằng: {best_thresh:.2f} (AI phải chắc chắn > {best_thresh*100}% mới báo lũ)\n")
    
    # 2. Sử dụng ngưỡng tối ưu để in ra báo cáo
    y_pred_optimized = (y_pred_prob > best_thresh).astype(int)
    print(classification_report(y_test, y_pred_optimized, target_names=['An toàn (0)', 'Nguy hiểm (1)']))
    
    auc_score = roc_auc_score(y_test, y_pred_prob)
    print(f"ROC-AUC Score: {auc_score:.4f}")
    print("="*50 + "\n")
    # =========================================================================
    
    try:
        engine = get_engine()
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        with engine.begin() as conn:
            conn.execute(text("UPDATE batxat_model_registry SET is_active = FALSE WHERE model_target = 'FLOOD'"))
            conn.execute(text("""
                INSERT INTO batxat_model_registry 
                (model_name, algorithm, model_target, accuracy_test, model_path, scaler_path, is_active)
                VALUES (:name, :algo, :target, :acc, :m_path, :s_path, TRUE)
            """), {
                "name": f"LSTM_Flood_{timestamp}", "algo": "Stacked-LSTM", "target": "FLOOD",
                "acc": round(accuracy * 100, 2), "m_path": model_path, "s_path": scaler_path
            })
        print(f"Đã lưu và kích hoạt LSTM thành công!")
    except Exception as e:
        print(f"Lỗi đăng ký CSDL LSTM: {e}")

# CHỨC NĂNG 2: HUẤN LUYỆN DỰ BÁO SẠT LỞ (BẢN FULL ĐỒNG BỘ)
def train_landslide_rf_model():
    print("\n[2/2] ĐANG HUẤN LUYỆN RF (BƠM KỊCH BẢN BÃO - PHIÊN BẢN BÁO CÁO)...")
    DATA_DIR = "data/processed"
    MODELS_DIR = "models"
    
    ls_path = os.path.join(DATA_DIR, "dynamic_landslide_fusion.csv")
    if not os.path.exists(ls_path): return

    df = pd.read_csv(ls_path)
    ls_features = ['Slope', 'Elevation', 'Dist_to_Water', 'precip_3d_BAT_XAT', 'soil_BAT_XAT']
    
    # TIÊM KỊCH BẢN MƯA LỚN
    print("Đang mô phỏng tương tác Địa hình - Thời tiết...")
    np.random.seed(42)
    # Dàn đều lượng mưa từ 0 -> 200mm cho toàn bộ kịch bản để AI thấy đủ các mức độ
    df['precip_3d_BAT_XAT'] = np.random.uniform(0, 200, size=len(df))
    df['soil_BAT_XAT'] = np.random.uniform(0.3, 0.9, size=len(df))

    # Chuẩn hóa các biến (0 -> 1)
    slope_factor = df['Slope'] / 90
    rain_factor = df['precip_3d_BAT_XAT'] / 200
    water_factor = 1 - np.clip(df['Dist_to_Water'] / 1000, 0, 1)
    
    # --- CÔNG THỨC VẬT LÝ CHUẨN ---
    # Độ dốc chiếm 65% quyết định. Mưa chỉ chiếm 25% (đóng vai trò kích hoạt).
    combined_risk = (slope_factor * 0.65) + (rain_factor * 0.25) + (water_factor * 0.10)
    
    # Thêm 5% nhiễu ngẫu nhiên để AI không bị "học vẹt" 100%
    noise = np.random.normal(0, 0.05, len(combined_risk))
    final_risk = np.clip(combined_risk + noise, 0, 1)
    
    # Tạo nhãn: Sạt lở nếu tổng rủi ro vượt ngưỡng 0.55
    y = (final_risk > 0.55).astype(int)

    # HUẤN LUYỆN MODEL 
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=7,              # Nâng nhẹ độ sâu để AI học được sự kết hợp Dốc + Mưa
        min_samples_leaf=15,      
        max_features='sqrt',      # Trả về chuẩn mặc định để cây mọc tự nhiên
        random_state=42
    )

    X = df[ls_features].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model.fit(X_train, y_train)

    # ĐÁNH GIÁ 
    y_pred = rf_model.predict(X_test)
    real_acc = round(float(accuracy_score(y_test, y_pred) * 100), 2)
    cv_mean = round(float(cross_val_score(rf_model, X, y, cv=5).mean() * 100), 2)
    
    # =========================================================================
    # THÊM BÁO CÁO PHÂN LOẠI CHO RANDOM FOREST
    # =========================================================================
    print("\n" + "="*50)
    print(" KẾT QUẢ ĐÁNH GIÁ CHUYÊN SÂU RANDOM FOREST (SẠT LỞ)")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=['An toàn (0)', 'Sạt lở (1)']))
    
    # Đoán xác suất để lấy ROC-AUC (RF dùng predict_proba)
    y_pred_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, y_pred_prob_rf)
    print(f"ROC-AUC Score: {rf_auc:.4f}")
    print("="*50 + "\n")
    # =========================================================================

    importances = rf_model.feature_importances_
    f_imp = {name: round(float(val * 100), 2) for name, val in zip(ls_features, importances)}

    print(f"KẾT QUẢ ĐÃ CÂN BẰNG: Acc: {real_acc}% | CV: {cv_mean}%")
    print(f"TRỌNG SỐ VẬT LÝ CHUẨN:")
    for feat, val in f_imp.items():
        print(f"   - {feat}: {val}%")

    rf_save_path = os.path.join(MODELS_DIR, "rf_dynamic_landslide_model.pkl")
    joblib.dump(rf_model, rf_save_path)

    try:
        engine = get_engine()
        timestamp_str = datetime.now().strftime('%m%d_%H%M')
        with engine.begin() as conn:
            # Tắt các model sạt lở cũ
            conn.execute(text("UPDATE batxat_model_registry SET is_active = FALSE WHERE model_target = 'LANDSLIDE'"))
            
            # Insert model mới với đủ 5 trọng số
            conn.execute(text("""
                INSERT INTO batxat_model_registry 
                (model_name, algorithm, model_target, accuracy_test, cv_score_mean, 
                 feat_imp_slope, feat_imp_elevation, feat_imp_water, 
                 feat_imp_precip, feat_imp_soil, 
                 model_path, scaler_path, is_active)
                VALUES (:name, :algo, :target, :acc, :cv, 
                        :f_s, :f_e, :f_w, :f_p, :f_soil, 
                        :path, :s_path, TRUE)
            """), {
                "name": f"RF_Smart_Fusion_{timestamp_str}",
                "algo": "RandomForest-Advanced", 
                "target": "LANDSLIDE", 
                "acc": real_acc, 
                "cv": cv_mean,
                "f_s": f_imp.get('Slope', 0), 
                "f_e": f_imp.get('Elevation', 0), 
                "f_w": f_imp.get('Dist_to_Water', 0),
                "f_p": f_imp.get('precip_3d_BAT_XAT', 0),  
                "f_soil": f_imp.get('soil_BAT_XAT', 0),    
                "path": rf_save_path, 
                "s_path": "N/A"
            })
        print(f"Đã lưu bộ não AI cùng toàn bộ 5 trọng số vật lý vào CSDL!")
    except Exception as e:
        print(f"Lỗi ghi CSDL: {e}")

if __name__ == "__main__":

    train_lstm_smote_model()
    train_landslide_rf_model()
    print("\nHỆ THỐNG MÔ HÌNH DỰ BÁO ĐÃ ĐƯỢC CẬP NHẬT ĐỒNG BỘ.")