import networkx as nx
import osmnx as ox
import time
import random
import numpy as np
from pathlib import Path

GRAPH_OUT_PATH = Path(__file__).resolve().parents[2] / "models" / "baxat_mcdm_final.graphml"

def run_large_scale_benchmark(num_samples=10000):
    print(f"ĐANG TẢI ĐỒ THỊ VÀ TIỀN XỬ LÝ (Chuẩn bị chạy {num_samples} kịch bản)...")
    G = ox.load_graphml(GRAPH_OUT_PATH)
    
    # Ép kiểu dữ liệu (tránh lỗi String)
    for u, v, k, data in G.edges(keys=True, data=True):
        data['length'] = float(data.get('length', 1.0))
        data['cost_speed'] = float(data.get('cost_speed', data['length']))
        data['cost_safety'] = float(data.get('cost_safety', data['length']))

    nodes = list(G.nodes())
    random.seed(42) # Giữ seed để kết quả ra giống nhau nếu hội đồng bắt chạy lại
    
    print(f"\nBẮT ĐẦU BENCHMARK 500 CẶP O-D NGẪU NHIÊN...")
    
    stats = {
        'rescue_dev': [], 'safety_dev': [],
        'short_time': [], 'rescue_time': [], 'safety_time': []
    }
    
    valid_count = 0
    while valid_count < num_samples:
        origin = random.choice(nodes)
        destination = random.choice(nodes)
        
        try:
            # 1. Dijkstra Gốc
            t0 = time.perf_counter()
            path_short = nx.shortest_path(G, origin, destination, weight='length')
            t_short = (time.perf_counter() - t0) * 1000
            
            # Lọc bỏ các đường quá ngắn (dưới 5 nodes) để kết quả phân tích có ý nghĩa
            if len(path_short) < 5: 
                continue
                
            len_short = sum(min([float(d.get('length', 0)) for d in G.get_edge_data(path_short[i], path_short[i+1]).values()]) for i in range(len(path_short)-1))
            
            # 2. Rescue (Cứu hộ)
            t0 = time.perf_counter()
            path_rescue = nx.shortest_path(G, origin, destination, weight='cost_speed')
            t_rescue = (time.perf_counter() - t0) * 1000
            len_rescue = sum(min([float(d.get('length', 0)) for d in G.get_edge_data(path_rescue[i], path_rescue[i+1]).values()]) for i in range(len(path_rescue)-1))
            
            # 3. Safety (An toàn)
            t0 = time.perf_counter()
            path_safety = nx.shortest_path(G, origin, destination, weight='cost_safety')
            t_safety = (time.perf_counter() - t0) * 1000
            len_safety = sum(min([float(d.get('length', 0)) for d in G.get_edge_data(path_safety[i], path_safety[i+1]).values()]) for i in range(len(path_safety)-1))
            
            # Tính phần trăm lệch
            dev_rescue = ((len_rescue - len_short) / len_short) * 100 if len_short > 0 else 0
            dev_safety = ((len_safety - len_short) / len_short) * 100 if len_short > 0 else 0
            
            stats['rescue_dev'].append(dev_rescue)
            stats['safety_dev'].append(dev_safety)
            stats['short_time'].append(t_short)
            stats['rescue_time'].append(t_rescue)
            stats['safety_time'].append(t_safety)
            
            valid_count += 1
            if valid_count % 50 == 0:
                print(f"Đã hoàn thành: {valid_count}/{num_samples} kịch bản...")
                
        except nx.NetworkXNoPath:
            continue

    # IN KẾT QUẢ TỔNG HỢP RA MÀN HÌNH ĐỂ ĐIỀN VÀO BÁO CÁO
    print("\n" + "="*50)
    print(" KẾT QUẢ THỐNG KÊ DIỆN RỘNG (ĐIỀN VÀO BẢNG 5.5)")
    print("="*50)
    print(f"1. Độ lệch không gian trung bình (Rescue) : +{np.mean(stats['rescue_dev']):.2f} %")
    print(f"2. Độ lệch không gian trung bình (Safety) : +{np.mean(stats['safety_dev']):.2f} %")
    print(f"3. Độ lệch không gian cực đại (Rescue)    : +{np.max(stats['rescue_dev']):.2f} %")
    print(f"4. Độ lệch không gian cực đại (Safety)    : +{np.max(stats['safety_dev']):.2f} %")
    print("-" * 50)
    print(f"5. Thời gian xử lý trung bình (Shortest)  : {np.mean(stats['short_time']):.2f} ms")
    print(f"6. Thời gian xử lý trung bình (Rescue)    : {np.mean(stats['rescue_time']):.2f} ms")
    print(f"7. Thời gian xử lý trung bình (Safety)    : {np.mean(stats['safety_time']):.2f} ms")
    print(f"8. Thời gian xử lý cực đại (Max Latency)  : {max(np.max(stats['short_time']), np.max(stats['rescue_time']), np.max(stats['safety_time'])):.2f} ms")
    print("="*50)

if __name__ == "__main__":
    run_large_scale_benchmark()