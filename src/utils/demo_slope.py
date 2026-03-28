import os
import sys
import networkx as nx
import osmnx as ox
import folium
import pandas as pd
import numpy as np
import warnings

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
warnings.filterwarnings('ignore')

def get_largest_component_safe(G):
    """Đảm bảo đồ thị liên thông tuyệt đối"""
    print("Đang chốt lại 'Lục địa chính' của mạng lưới giao thông...")
    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()

def visualize_baxat_final_system():
    print("BƯỚC 09: KHỞI TẠO DASHBOARD CỨU HỘ VÀ MÔ PHỎNG SẠT LỞ...")
    
    path_graph = os.path.join(project_root, "models", "baxat_mcdm_final.graphml")
    if not os.path.exists(path_graph):
        print(f"Lỗi: Không tìm thấy {path_graph}. Bạn cần chạy Bước 08 trước!")    
        return
        
    G = ox.load_graphml(path_graph)
    G = G.to_undirected()
    G = get_largest_component_safe(G)
    
    print("⚙️ Đang chuẩn hóa trọng số mạng lưới...")
    for u, v, k, data in G.edges(data=True, keys=True):
        data['length_m'] = float(data.get('length_m', data.get('length', 1))) 
        data['cost_safety'] = float(data.get('cost_safety', 1))
        data['cost_speed'] = float(data.get('cost_speed', 1))
        data['avg_slope'] = float(data.get('avg_slope', 0))
        data['avg_elevation'] = float(data.get('avg_elevation', 0))

    start_lat, start_lng = 22.465896, 103.818626
    end_lat, end_lng = 22.467614, 103.835163

    orig_node = ox.distance.nearest_nodes(G, X=start_lng, Y=start_lat)
    dest_node = ox.distance.nearest_nodes(G, X=end_lng, Y=end_lat)

    try:
        r_short = nx.shortest_path(G, orig_node, dest_node, weight='length_m')
    except nx.NetworkXNoPath:
        print("❌ Lỗi: Không có đường nối gốc. Kiểm tra lại tọa độ.")
        return

    print("🌩️ Đang mô phỏng một vụ sạt lở cắt ngang tuyến đường ngắn nhất...")
    mid_index = len(r_short) // 2
    u_block, v_block = r_short[mid_index], r_short[mid_index + 1]
    
    landslide_lat = (G.nodes[u_block]['y'] + G.nodes[v_block]['y']) / 2
    landslide_lng = (G.nodes[u_block]['x'] + G.nodes[v_block]['x']) / 2

    for k in G[u_block][v_block]:
        G[u_block][v_block][k]['cost_safety'] = 9999999
        G[u_block][v_block][k]['cost_speed'] = 9999999
        G[u_block][v_block][k]['avg_slope'] = 90  

    try:
        r_safe  = nx.shortest_path(G, orig_node, dest_node, weight='cost_safety')
        r_rescue = nx.shortest_path(G, orig_node, dest_node, weight='cost_speed')
    except nx.NetworkXNoPath:
        print("Cảnh báo: Điểm sạt lở đã cô lập hoàn toàn khu vực, không còn đường vòng!")
        return

    def analyze_route_metrics(route, name):
        dist, danger_segs = 0, 0
        slopes, elevs = [], []
        
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            data = G.get_edge_data(u, v)[0]
            
            s = data.get('avg_slope', 0)
            dist += data.get('length_m', 0)
            slopes.append(s)
            elevs.append(data.get('avg_elevation', 0))
            if s > 25: danger_segs += 1
            
        return {
            "Phương án": name,
            "Quãng đường (km)": round(dist / 1000, 2),
            "Độ dốc TB (°)": round(np.mean(slopes), 1) if slopes else 0,
            "Độ dốc Max (°)": round(np.max(slopes), 1) if slopes else 0,
            "Đoạn nguy hiểm (>25°)": danger_segs,
        }

    df_compare = pd.DataFrame([
        analyze_route_metrics(r_short, "Đường Cũ (Đã bị chặn)"),
        analyze_route_metrics(r_safe, "Đường AN TOÀN (Vòng qua rủi ro)"),
        analyze_route_metrics(r_rescue, "Đường CỨU HỘ (Ưu tiên xe cơ giới)")
    ])

    print("\n" + "═"*75)
    print("BẢNG SO SÁNH ĐỊNH LƯỢNG KHI XẢY RA SẠT LỞ")
    print("═"*75)
    print(df_compare.to_string(index=False))
    print("═"*75)
    
    results_dir = os.path.join(project_root, "results")
    df_compare.to_csv(os.path.join(results_dir, "route_comparison_report.csv"), index=False, encoding='utf-8-sig')

    m = folium.Map(location=[start_lat, start_lng], zoom_start=15, tiles='CartoDB dark_matter')

    # Marker Vị trí Sạt lở
    folium.Marker(
        [landslide_lat, landslide_lng], 
        popup="<b>ĐIỂM SẠT LỞ KHẨN CẤP</b>", 
        icon=folium.Icon(color='red', icon='fire')
    ).add_to(m)
    folium.Circle([landslide_lat, landslide_lng], radius=80, color='red', fill=True, fill_opacity=0.4).add_to(m)

    def plot_route(route, color, name, weight, dash_array=None):
        coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
        folium.PolyLine(coords, color=color, weight=weight, opacity=0.8, 
                        popup=name, dash_array=dash_array).add_to(m)

    plot_route(r_short, '#ff4444', 'Đường ngắn nhất (Bị đứt gãy)', 4, dash_array='10, 10')
    
    plot_route(r_safe, '#00FF00', 'Đường AN TOÀN (AI Đề xuất)', 8) 
    plot_route(r_rescue, '#00A8FF', 'Đường CỨU HỘ', 5)

    folium.Marker([start_lat, start_lng], popup="Trạm Cứu Hộ", icon=folium.Icon(color='blue', icon='play')).add_to(m)
    folium.Marker([end_lat, end_lng], popup="Điểm Cô Lập", icon=folium.Icon(color='green', icon='flag')).add_to(m)
    
    output_map = os.path.join(results_dir, "Demo_HoiDong_TimDuong_SatLo.html")
    m.save(output_map)
    print(f"Đã xuất bản đồ có mô phỏng sạt lở tại: {output_map}")

if __name__ == "__main__":
    visualize_baxat_final_system()