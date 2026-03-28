import osmnx as ox
import networkx as nx
import folium
import os
import pandas as pd
import numpy as np

def get_largest_component_safe(G):
    """Đảm bảo đồ thị liên thông tuyệt đối"""
    print("Đang chốt lại 'Lục địa chính' của mạng lưới giao thông...")
    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()

def visualize_baxat_final_system():
    print("BƯỚC 09: KHỞI TẠO DASHBOARD CỨU HỘ THÔNG MINH BÁT XÁT...")
    
    path_graph = "models/baxat_mcdm_final.graphml"
    if not os.path.exists(path_graph):
        print("Lỗi: Bạn cần chạy Bước 08 trước!")    
        return
        
    G = ox.load_graphml(path_graph)
    G = G.to_undirected()
    G = get_largest_component_safe(G)
    
    #Chuẩn hóa dữ liệu
    for u, v, k, data in G.edges(data=True, keys=True):
        data['length'] = float(data.get('mm_len', 0)) # Sử dụng mm_len từ các bước trước
        data['cost_safety'] = float(data.get('cost_safety', 0))
        data['cost_speed'] = float(data.get('cost_speed', 0))
        data['avg_slope'] = float(data.get('avg_slope', 0))
        data['avg_elevation'] = float(data.get('avg_elevation', 0))

    # Toạ độ điểm đi và điểm đến
    start_lat, start_lng = 22.465896, 103.818626
    end_lat, end_lng = 22.467614, 103.835163

    orig_node = ox.distance.nearest_nodes(G, X=start_lng, Y=start_lat)
    dest_node = ox.distance.nearest_nodes(G, X=end_lng, Y=end_lat)

    #Tính toán 3 lộ trình chiến lược
    try:
        r_short = nx.shortest_path(G, orig_node, dest_node, weight='length')
        r_safe  = nx.shortest_path(G, orig_node, dest_node, weight='cost_safety')
        r_rescue = nx.shortest_path(G, orig_node, dest_node, weight='cost_speed')
        print("Đã tính toán xong 3 lộ trình chiến lược.")
    except nx.NetworkXNoPath:
        print("Lỗi: Vẫn không có đường nối! Kiểm tra lại tính liên thông của Shapefile.")
        return

    def analyze_route_metrics(route, name):
        dist = 0
        slopes = []
        elevs = []
        danger_segs = 0
        
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            # Lấy data cạnh đầu tiên (do đã undirected)
            data = G.get_edge_data(u, v)[0]
            
            d = data.get('length', 0)
            s = data.get('avg_slope', 0)
            dist += d
            slopes.append(s)
            elevs.append(data.get('avg_elevation', 0))
            if s > 25: danger_segs += 1
            
        return {
            "Phương án": name,
            "Quãng đường (km)": round(dist / 1000, 2),
            "Độ dốc TB (°)": round(np.mean(slopes), 1),
            "Độ dốc Max (°)": round(np.max(slopes), 1),
            "Đoạn nguy hiểm (>25°)": danger_segs,
            "Cao độ TB (m)": round(np.mean(elevs), 0)
        }

    results = [
        analyze_route_metrics(r_short, "Ngắn nhất (Rủi ro)"),
        analyze_route_metrics(r_safe, "AN TOÀN (Né thiên tai)"),
        analyze_route_metrics(r_rescue, "CỨU HỘ (Ưu tiên tốc độ)")
    ]
    df_compare = pd.DataFrame(results)

    print("\n" + "═"*75)
    print("📊 BẢNG SO SÁNH ĐỊNH LƯỢNG CÁC LỘ TRÌNH CỨU HỘ")
    print("═"*75)
    print(df_compare.to_string(index=False))
    print("═"*75)
    
    if not os.path.exists("results"): os.makedirs("results")
    df_compare.to_csv("results/route_comparison_report.csv", index=False, encoding='utf-8-sig')

    m = folium.Map(location=[start_lat, start_lng], zoom_start=14)
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', 
        attr='Google Maps Satellite', name='Vệ tinh & Địa hình', overlay=False
    ).add_to(m)

    # Tô màu cảnh báo dốc
    for u, v, k, data in G.edges(data=True, keys=True):
        slope = float(data.get('avg_slope', 0))
        if slope > 20:
            coords = [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
            folium.PolyLine(coords, color='orange', weight=2, opacity=0.3).add_to(m)

    #Vẽ 3 lộ trình
    def plot_route(route, color, name, weight):
        coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
        folium.PolyLine(coords, color=color, weight=weight, opacity=0.8, popup=name).add_to(m)

    plot_route(r_short, 'red', 'Đường ngắn nhất', 4)
    plot_route(r_safe, '#00FF00', 'Đường AN TOÀN', 9) # Xanh lá đậm, nét dày nhất
    plot_route(r_rescue, 'blue', 'Đường CỨU HỘ', 6)

    # Marker
    folium.Marker([start_lat, start_lng], icon=folium.Icon(color='blue', icon='person', prefix='fa')).add_to(m)
    folium.Marker([end_lat, end_lng], icon=folium.Icon(color='green', icon='hospital', prefix='fa')).add_to(m)
    
    folium.LayerControl().add_to(m)
    m.save("results/baxat_final_map_v2.html")
    print(f"Đã xuất bản đồ và báo cáo so sánh vào thư mục /results.")
    
if __name__ == "__main__":
    visualize_baxat_final_system()