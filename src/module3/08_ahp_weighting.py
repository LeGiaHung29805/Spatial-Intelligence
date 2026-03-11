import osmnx as ox
import pandas as pd
import numpy as np
import os

def calculate_ahp_6x6(strategy="safety"):
    """
    Ma trận so sánh cặp AHP 6x6.
    [C1: Khoảng cách, C2: Ngập, C3: Sạt lở, C4: Hạng đường, C5: Cầu, C6: Báo cáo]
    """
    if strategy == "safety":
        # Ưu tiên né tránh Sạt lở (C3) và Ngập (C2) tuyệt đối
        matrix = np.array([
            [1,   1/7, 1/9, 1/5, 1/6, 1/5], # C1
            [7,   1,   1/2, 3,   2,   2],   # C2
            [9,   2,   1,   5,   3,   3],   # C3
            [5,   1/3, 1/5, 1,   1/2, 1/2], # C4
            [6,   1/2, 1/3, 2,   1,   1],   # C5
            [5,   1/2, 1/3, 2,   1,   1]    # C6
        ])
    else: # strategy == "rescue"
        # Ưu tiên đường lớn (C4) để xe cơ giới cứu hộ di chuyển nhanh
        matrix = np.array([
            [1,   2,   1/3, 1/4, 1/2, 1],   # C1
            [1/2, 1,   1/4, 1/5, 1/3, 1/2], # C2
            [3,   4,   1,   1,   2,   2],   # C3
            [4,   5,   1,   1,   2,   3],   # C4
            [2,   3,   1/2, 1/2, 1,   1],   # C5
            [1,   2,   1/2, 1/3, 1,   1]    # C6
        ])
    
    # Tính vector trọng số (Eigenvector)
    weights = np.mean(matrix / np.sum(matrix, axis=0), axis=1)
    return weights

def apply_mcdm_to_raw_graph():
    print("ĐANG TÍCH HỢP TRÍ TUỆ AHP VÀO ĐỒ THỊ NGUYÊN BẢN (4326)...")
    
    # Đường dẫn file
    graph_in = "data/shapefiles/infrastructure/Duong_Bat_Xat_Connected.graphml"
    risk_path = "data/processed/daily_risk_report.csv"
    output_path = "emodels/baxat_mcdm_final.graphml"
    
    if not os.path.exists(graph_in):
        print("Không tìm thấy file đồ thị gốc! Hãy chạy lại file Download bản nguyên bản.")
        return

    # 1. Nạp đồ thị
    G = ox.load_graphml(graph_in)
    
    # 2. Đọc rủi ro dự báo từ Module 2
    try:
        risk_df = pd.read_csv(risk_path)
        f_risk = float(risk_df['flood_risk'].iloc[0]) / 100
        l_risk = float(risk_df['landslide_risk'].iloc[0]) / 100
    except:
        print("Cảnh báo: Thiếu file rủi ro, sử dụng giá trị mặc định (50%).")
        f_risk, l_risk = 0.5, 0.5

    # 3. Tính trọng số AHP
    w_s = calculate_ahp_6x6("safety")
    w_sp = calculate_ahp_6x6("rescue")

    # 4. GÁN TRỌNG SỐ CHO TỪNG ĐOẠN ĐƯỜNG
    for u, v, k, data in G.edges(data=True, keys=True):
        # Lấy chiều dài mét (OSMnx tự tính toán khi tải về)
        L = float(data.get('length', 100.0))
        
        # Nhận diện Hạng đường (C4) từ Tag 'highway' của OSM
        h_way = str(data.get('highway', 'unclassified')).lower()
        if any(x in h_way for x in ['primary', 'secondary', 'trunk', 'quốc lộ']):
            rc = 1.0 # Đường tốt
        elif 'tertiary' in h_way:
            rc = 2.0 # Đường trung bình
        else:
            rc = 4.0 # Đường mòn, ngõ cụt

        # Nhận diện Cầu (C5)
        is_br = 1.0 if (data.get('bridge') != 'no' and 'bridge' in data) else 0.0
        
        # Mặc định phản hồi cộng đồng (C6)
        cr = 0.0 

        # --- CHUẨN HÓA & TÍNH TOÁN MCDM ---
        # C1: Chiều dài thực tế (m)
        # C2, C3: Khuếch đại rủi ro thiên tai để AI biết đường né
        c = np.array([
            L,                # C1
            f_risk * 5000,    # C2
            l_risk * 7000,    # C3
            rc * 500,         # C4
            is_br * 2000,     # C5
            cr * 10000        # C6
        ])

        # Nhúng kết quả vào thuộc tính cạnh để Module 3 tìm đường
        data['cost_safety'] = float(np.dot(w_s, c))
        data['cost_speed'] = float(np.dot(w_sp, c))

    # 5. LƯU ĐỒ THỊ THÔNG MINH
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        
    ox.save_graphml(G, output_path)
    
    print("-" * 50)
    print(f"THÀNH CÔNG: Đồ thị thông minh đã sẵn sàng cho Module 3.")
    print(f"Trọng số Safety (C1->C6): {np.round(w_s, 3)}")
    print(f"Trọng số Rescue (C1->C6): {np.round(w_sp, 3)}")
    print(f"File: {output_path}")
    print("-" * 50)

if __name__ == "__main__":
    apply_mcdm_to_raw_graph()