import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_landslide_model():
    print("KHỞI ĐỘNG BƯỚC 2: HUẤN LUYỆN AI (RANDOM FOREST)...")

    data_path = "data/processed/landslide_training_data.csv"
    model_dir = "models"
    model_path = os.path.join(model_dir, "rf_landslide_model.pkl")

    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Lỗi: Không tìm thấy file dữ liệu {data_path}")
        return

    # 1. Đọc "Sách giáo khoa" (File CSV từ Bước 01)
    print("Đang nạp kho dữ liệu huấn luyện...")
    df = pd.read_csv(data_path)
    
    # 2. Tách Dữ liệu: Bài tập (X) và Đáp án (y)
    X = df[['Elevation_Norm', 'Slope_Norm']] # AI chỉ học trên thang đo [0,1]
    y = df['Landslide_Label']

    # 3. Chia tập dữ liệu: 80% để Học, 20% để Thi
    print("Đang chia dữ liệu: 80% để Học, 20% để Thi...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Bắt đầu quá trình HỌC (Training)
    print("Đang trồng Rừng ngẫu nhiên (Random Forest) với 100 cây quyết định...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Lệnh .fit() chính là lúc AI động não tìm quy luật
    rf_model.fit(X_train, y_train)

    # 5. Làm bài kiểm tra và Chấm điểm
    print("Đang chấm điểm bài thi của AI trên 200 điểm bị giấu...")
    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("\n" + "="*50)
    print(f"ĐỘ CHÍNH XÁC TỔNG THỂ (ACCURACY): {accuracy * 100:.2f}%")
    print("="*50)
    
    print("\nYẾU TỐ QUYẾT ĐỊNH (FEATURE IMPORTANCE):")
    importances = rf_model.feature_importances_
    print(f"   - Cao độ (Elevation): {importances[0] * 100:.1f}%")
    print(f"   - Độ dốc (Slope): {importances[1] * 100:.1f}%")

    # 6. Xuất xưởng "Bộ não"
    print("\nĐang đóng gói và lưu 'Bộ não' AI...")
    joblib.dump(rf_model, model_path)
    print(f"Hoàn thành! Model đã được lưu an toàn tại: {model_path}")

if __name__ == "__main__":
    train_landslide_model()