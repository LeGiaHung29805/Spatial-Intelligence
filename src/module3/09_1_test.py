import osmnx as ox
import networkx as nx
import folium
from folium import plugins
import os

# ==========================================================
# 1. ĐỊNH NGHĨA CÁC HÀM BỔ TRỢ (ĐỂ SỬA LỖI NAMEERROR)
# ==========================================================

def get_largest_component_safe(G):
    """Đảm bảo đồ thị liên thông để không bao giờ bị lỗi 'No Path'"""
    if hasattr(ox, 'largest_component'):
        return ox.largest_component(G)
    # Cách dự phòng dùng NetworkX thuần
    largest_cc = max(nx.weakly_connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()

def get_route_dist_raw(G, route):
    """Tính tổng chiều dài thực tế (mét) của tuyến đường từ thuộc tính 'length'"""
    total_dist = 0
    for u, v in zip(route[:-1], route[1:]):
        # Lấy dữ liệu cạnh (mặc định key=0 cho đồ thị OSM)
        data = G.get_edge_data(u, v, 0)
        if data:
            total_dist += float(data.get('length', 0))
    return total_dist

# ==========================================================
# 2. HÀM CHÍNH: TRỰC QUAN HÓA
# ==========================================================

def visualize_baxat_decision_map():
    print("🎨 ĐANG KHỞI TẠO BẢN ĐỒ 3 CHIẾN LƯỢC: NGẮN NHẤT - AN TOÀN - CỨU HỘ...")
    
    path_graph = "models/baxat_mcdm_final.graphml"
    G = ox.load_graphml(path_graph)
    
    # Chuẩn hóa dữ liệu
    for u, v, k, data in G.edges(data=True, keys=True):
        data['length'] = float(data.get('length', 0))
        data['cost_safety'] = float(data.get('cost_safety', 0))
        data['cost_speed'] = float(data.get('cost_speed', 0))
    
    G = get_largest_component_safe(G)

    # 📍 Tọa độ Start/End của bạn
    start_lat, start_lng = 22.552237, 103.871800 
    end_lat, end_lng = 22.546029554286292, 103.88501551955392 

    # --- ⚠️ GIẢ LẬP 2 ĐIỂM SẠT LỞ ---
    landslide_blocks = [(22.548061, 103.876200), (22.550878, 103.878643)]
    for b_lat, b_lng in landslide_blocks:
        b_node = ox.distance.nearest_nodes(G, X=b_lng, Y=b_lat)
        for u, v, k, data in G.edges(b_node, data=True, keys=True):
            data['cost_safety'] = 10**9 
            data['cost_speed'] = 10**9

    # --- 🚀 TÌM ĐƯỜNG THEO 3 TRỌNG SỐ KHÁC NHAU ---
    orig_node = ox.distance.nearest_nodes(G, X=start_lng, Y=start_lat)
    dest_node = ox.distance.nearest_nodes(G, X=end_lng, Y=end_lat)

    try:
        # 1. Ngắn nhất (Weight: length)
        route_shortest = nx.shortest_path(G, orig_node, dest_node, weight='length')
        # 2. An toàn dân sự (Weight: cost_safety)
        route_safety = nx.shortest_path(G, orig_node, dest_node, weight='cost_safety')
        # 3. Cứu hộ cơ giới (Weight: cost_speed) -> ĐƯỜNG THỨ 3 ĐÂY
        route_rescue = nx.shortest_path(G, orig_node, dest_node, weight='cost_speed')

        print("-" * 50)
        print(f"🚩 Tuyến NGẮN NHẤT (Đỏ): {get_route_dist_raw(G, route_shortest):.1f} m")
        print(f"🛡️ Tuyến AN TOÀN (Xanh lá): {get_route_dist_raw(G, route_safety):.1f} m")
        print(f"🚑 Tuyến CỨU HỘ (Xanh dương): {get_route_dist_raw(G, route_rescue):.1f} m")
        print("-" * 50)

        # --- VẼ BẢN ĐỒ ---
        m = folium.Map(location=[start_lat, start_lng], zoom_start=15)
        folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr='Google', name='Vệ tinh').add_to(m)

        def plot_path(route, color, weight, label, opacity=0.7):
            coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
            folium.PolyLine(coords, color=color, weight=weight, opacity=opacity, tooltip=label).add_to(m)

        # Vẽ 3 đường với độ dày khác nhau để không đè mất nhau
        plot_path(route_safety, 'green', 12, 'Safety Route (Dân sự)', 0.4) # Dày nhất, nhạt nhất
        plot_path(route_rescue, 'blue', 7, 'Rescue Route (Xe cơ giới)', 0.8) # Vừa
        plot_path(route_shortest, 'red', 3, 'Shortest Path (Rủi ro)', 1.0)  # Mỏng nhất, đậm nhất

        # Marker sạt lở và Start/End
        for b_lat, b_lng in landslide_blocks:
            folium.Marker([b_lat, b_lng], icon=folium.Icon(color='red', icon='remove-sign')).add_to(m)
        folium.Marker([start_lat, start_lng], popup="START", icon=folium.Icon(color='blue')).add_to(m)
        folium.Marker([end_lat, end_lng], popup="END", icon=folium.Icon(color='green')).add_to(m)

        m.save("results/baxat_3_routes_test.html")
        print("✅ Thành công! Bạn đã có đủ 3 kịch bản đường đi.")

    except nx.NetworkXNoPath:
        print("❌ Lỗi: Không tìm thấy đường!")

if __name__ == "__main__":
    visualize_baxat_decision_map()