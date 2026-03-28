import joblib
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

def simulate_storm(luong_mua_mm):
    print(f"\nKÍCH HOẠT TRẠM MÔ PHỎNG THỜI TIẾT: Bơm {luong_mua_mm}mm mưa vào hệ thống...")
    
    model_path = "models/rf_dynamic_landslide_model.pkl"
    if not os.path.exists(model_path):
        print("Lỗi: Không tìm thấy model AI. Hãy chạy lại Bước 05!")
        return
        
    rf_model = joblib.load(model_path)

    do_am_dat = 0.9 if luong_mua_mm >= 150 else 0.4
    
    # ['Slope', 'Elevation', 'Dist_to_Water', 'precip_3d_BAT_XAT', 'soil_BAT_XAT']
    scenario_data = pd.DataFrame({
        'Slope': [35],              # Dốc 35 độ (Điểm tới hạn)
        'Elevation': [800],         # Cao độ 800m
        'Dist_to_Water': [100],     # Cách suối 100m
        'precip_3d_BAT_XAT': [luong_mua_mm], 
        'soil_BAT_XAT': [do_am_dat]
    })

    ai_prediction = rf_model.predict_proba(scenario_data)[0]
    landslide_risk_percent = round(ai_prediction[1] * 100, 2)
    

    flood_risk_percent = min(100.0, luong_mua_mm / 200 * 100)

    print("\n" + "═"*50)
    print("KẾT QUẢ PHÂN TÍCH TỪ BỘ NÃO TRÍ TUỆ NHÂN TẠO")
    print("═"*50)
    print(f"Lượng mưa đầu vào: {luong_mua_mm} mm")
    print(f"Tọa độ thử nghiệm: Dốc 35°, Ven suối")
    print(f"XÁC SUẤT SẠT LỞ (AI Phán đoán): {landslide_risk_percent}%")
    print(f"XÁC SUẤT NGẬP LỤT: {flood_risk_percent}%")
    print("═"*50)

    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "daily_risk_report.csv")
    
    report_df = pd.DataFrame({
        'date': [pd.Timestamp.now().strftime("%Y-%m-%d")],
        'rainfall_mm': [luong_mua_mm],
        'landslide_risk': [landslide_risk_percent],
        'flood_risk': [flood_risk_percent]
    })
    
    report_df.to_csv(report_path, index=False)
    print(f"Đã xuất báo cáo rủi ro ra file: {report_path}")
    print("SẴN SÀNG: Bây giờ hãy chạy BƯỚC 08 và BƯỚC 10 để xem đường đi biến đổi!\n")

if __name__ == "__main__":

    LƯỢNG_MƯA_HÔM_NAY = 100 
    simulate_storm(LƯỢNG_MƯA_HÔM_NAY)
