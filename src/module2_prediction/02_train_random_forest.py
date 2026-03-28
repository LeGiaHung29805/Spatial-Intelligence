import pandas as pd
import numpy as np
import os
import joblib
import sys
from sqlalchemy import text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- CẤU HÌNH ĐƯỜNG DẪN ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
csdl_path = os.path.join(src_dir, 'CSDL')

if csdl_path not in sys.path: sys.path.append(csdl_path)
if src_dir not in sys.path: sys.path.append(src_dir)

try:
    from config.db_config import get_engine
except ModuleNotFoundError:
    print("Lỗi: Không tìm thấy module cấu hình CSDL!")
    sys.exit(1)
    
def train_landslide_model_enhanced():
    print("KHỞI ĐỘNG BƯỚC 2: HUẤN LUYỆN AI 5 TRỌNG SỐ (PHYSICS-BASED)...")

    data_path = "data/processed/landslide_training_data.csv"
    model_dir = "models"
    model_path = os.path.join(model_dir, "rf_landslide_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    if not os.path.exists(model_dir): os.makedirs(model_dir)
    if not os.path.exists(data_path):
        print(f"Lỗi: Không tìm thấy file {data_path}"); return

    df = pd.read_csv(data_path)
    
    features = ['Slope', 'Elevation', 'Dist_to_Water', 'precip_3d_BAT_XAT', 'soil_BAT_XAT']

    if 'precip_3d_BAT_XAT' not in df.columns:
        print(" Đang tiêm kịch bản Mưa và Đất vào dữ liệu huấn luyện...")
        np.random.seed(42)
        df['precip_3d_BAT_XAT'] = np.random.uniform(0, 250, size=len(df))
        df['soil_BAT_XAT'] = np.random.uniform(0.3, 0.9, size=len(df))

    X = df[features].fillna(0)

    print("Áp dụng trọng số vật lý: Dốc (60%) + Mưa (25%) + Nước (10%) + Đất (5%)...")
    slope_f = X['Slope'] / 90
    rain_f = X['precip_3d_BAT_XAT'] / 250
    water_f = 1 - np.clip(X['Dist_to_Water'] / 1000, 0, 1)
    risk = (slope_f * 0.60) + (rain_f * 0.25) + (water_f * 0.10) + (X['soil_BAT_XAT'] * 0.05)
    y = (risk > 0.55).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, scaler_path)
    print(f"Đã lưu Thước đo: {scaler_path}")

    #HUẤN LUYỆN VÀ LƯU MODEL
    print("Đang huấn luyện Random Forest (5 features)...")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    #rf_landslide_model.pkl (Bộ não AI)
    joblib.dump(rf_model, model_path)
    print(f"Đã lưu Mô hình: {model_path}")

    accuracy = accuracy_score(y_test, rf_model.predict(X_test_scaled))
    imp = rf_model.feature_importances_

    print(f"\n{'='*30}\nĐỘ CHÍNH XÁC: {accuracy*100:.2f}%\n{'='*30}")

    # try:
    #     engine = get_engine()
    #     with engine.begin() as conn:
    #         conn.execute(text("UPDATE batxat_model_registry SET is_active = FALSE WHERE model_target = 'LANDSLIDE'"))
    #         query = text("""
    #             INSERT INTO batxat_model_registry 
    #             (model_name, algorithm, model_target, accuracy_test, 
    #              feat_imp_slope, feat_imp_elevation, feat_imp_water, feat_imp_precip, feat_imp_soil,
    #              model_path, scaler_path, is_active)
    #             VALUES 
    #             (:m_name, :algo, :target, :acc, :f_s, :f_e, :f_w, :f_p, :f_soil, :m_path, :s_path, TRUE)
    #         """)
    #         conn.execute(query, {
    #             "m_name": "RF_5Features_Dynamic", "algo": "RandomForest", "target": "LANDSLIDE",
    #             "acc": round(float(accuracy * 100), 2),
    #             "f_s": round(float(imp[0]*100), 2), "f_e": round(float(imp[1]*100), 2),
    #             "f_w": round(float(imp[2]*100), 2), "f_p": round(float(imp[3]*100), 2),
    #             "f_soil": round(float(imp[4]*100), 2),
    #             "m_path": model_path, "s_p": scaler_path # Fix key s_path
    #         })
    #     print("Đã đăng ký Model 5 trọng số vào hệ thống!")
    # except Exception as e:
    #     print(f"Lỗi CSDL: {e}")

if __name__ == "__main__":
    train_landslide_model_enhanced()