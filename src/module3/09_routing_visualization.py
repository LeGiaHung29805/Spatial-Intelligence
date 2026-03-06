import osmnx as ox
import networkx as nx
import folium
from folium import plugins
import os

def get_largest_component_safe(G):
    """Đảm bảo đồ thị liên thông để không bao giờ bị lỗi 'No Path'"""
    if hasattr(ox, 'largest_component'):
        return ox.largest_component(G)
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

def visualize_baxat_decision_map():
    print("🎨 ĐANG KHỞI TẠO BẢN ĐỒ CỨU HỘ THÔNG MINH BÁT XÁT...")
    
    path_graph = "models/baxat_mcdm_final.graphml"
    if not os.path.exists(path_graph):
        print("❌ Lỗi: Bạn cần chạy Bước 08 trước để có dữ liệu AHP!")
        return
        
    G = ox.load_graphml(path_graph)
    
    # 🔥 BƯỚC SỬA LỖI QUAN TRỌNG: ÉP KIỂU DỮ LIỆU SỐ CHO TẤT CẢ CẠNH
    print("🧹 Đang chuẩn hóa dữ liệu số (Fixing TypeError)...")
    for u, v, k, data in G.edges(data=True, keys=True):
        # Ép tất cả trọng số về float để Dijkstra có thể cộng dồn
        data['length'] = float(data.get('length', 0))
        data['cost_safety'] = float(data.get('cost_safety', 0))
        data['cost_speed'] = float(data.get('cost_speed', 0))
    
    # 1. Lọc mạng lưới liên thông (Giữ nguyên phần này)
    G = get_largest_component_safe(G)
    print(f"✅ Mạng lưới liên thông sẵn sàng: {len(G.nodes)} nút giao.")

    # 2. TOẠ ĐỘ ĐIỂM ĐI VÀ ĐIỂM ĐẾN (Ví dụ tại khu vực trung tâm hoặc điểm dân cư)
    # Gợi ý: Bạn có thể thay đổi tọa độ này để kiểm tra các khu vực sạt lở khác nhau
    start_lat, start_lng = 22.552237, 103.871800 # Điểm xuất phát
    end_lat, end_lng = 22.546029554286292, 103.88501551955392  # Điểm cứu trợ / Trạm y tế

    # Tìm Node gần nhất trên đồ thị dựa vào GPS
    orig_node = ox.distance.nearest_nodes(G, X=start_lng, Y=start_lat)
    dest_node = ox.distance.nearest_nodes(G, X=end_lng, Y=end_lat)
    
    print(f"📍 Đã bắt được Node gần nhất: Start({orig_node}) -> End({dest_node})")

    # 3. THỰC THI THUẬT TOÁN TÌM ĐƯỜNG ĐA CHIẾN LƯỢC
    print("🚀 AI đang tính toán lộ trình tối ưu dựa trên trọng số AHP...")
    try:
        # Lộ trình ngắn nhất (Chỉ quan tâm khoảng cách - Rủi ro cao)
        route_shortest = nx.shortest_path(G, orig_node, dest_node, weight='length')
        
        # Lộ trình AN TOÀN (Ưu tiên né sạt lở/ngập lụt tuyệt đối)
        route_safety = nx.shortest_path(G, orig_node, dest_node, weight='cost_safety')
        
        # Lộ trình CỨU HỘ (Cân bằng giữa tốc độ và khả năng xe cơ giới di chuyển)
        route_rescue = nx.shortest_path(G, orig_node, dest_node, weight='cost_speed')

        # --- PHÂN TÍCH QUÃNG ĐƯỜNG ---
        d_short = get_route_dist_raw(G, route_shortest)
        d_safety = get_route_dist_raw(G, route_safety)
        d_rescue = get_route_dist_raw(G, route_rescue)

        print("-" * 50)
        print(f"📊 BÁO CÁO PHÂN TÍCH LỘ TRÌNH:")
        print(f"🚩 Tuyến NGẮN NHẤT: {d_short:.1f} m (Cảnh báo: Có thể đi qua vùng sạt lở)")
        print(f"🛡️ Tuyến AN TOÀN:   {d_safety:.1f} m (Né tránh rủi ro sạt lở)")
        print(f"🚑 Tuyến CỨU HỘ:   {d_rescue:.1f} m (Phù hợp xe cơ giới)")
        print("-" * 50)

    except nx.NetworkXNoPath:
        print("❌ Lỗi: Không thể tìm thấy đường nối giữa 2 khu vực này (Kẽ hở địa lý quá lớn)!")
        return

    # 4. VẼ BẢN ĐỒ TƯƠNG TÁC FOLIUM
    # Sử dụng Google Satellite để nhìn rõ địa hình đồi núi Bát Xát
    m = folium.Map(location=[start_lat, start_lng], zoom_start=15)
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google',
        name='Bản đồ Vệ tinh',
        overlay=False,
        control=True
    ).add_to(m)
    folium.TileLayer('OpenStreetMap', name='Bản đồ Giao thông').add_to(m)

    # Hàm vẽ tuyến đường
    def plot_path(route, color, label, weight_line):
        coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
        folium.PolyLine(coords, color=color, weight=weight_line, opacity=0.8, tooltip=label).add_to(m)

    # Vẽ 3 tuyến đường với độ dày khác nhau để phân biệt
    plot_path(route_shortest, 'red', 'Shortest Path (Nguy hiểm)', 4)
    plot_path(route_safety, 'green', 'Safety Route (An toàn dân sự)', 7)
    plot_path(route_rescue, 'blue', 'Rescue Route (Lực lượng chức năng)', 7)

    # Marker điểm đầu và điểm cuối
    folium.Marker([start_lat, start_lng], popup="VỊ TRÍ HIỆN TẠI", icon=folium.Icon(color='blue', icon='info-sign')).add_to(m)
    folium.Marker([end_lat, end_lng], popup="ĐIỂM ĐẾN AN TOÀN", icon=folium.Icon(color='green', icon='home')).add_to(m)

    # Thêm bảng điều khiển lớp (Layer Control)
    folium.LayerControl().add_to(m)

    # 5. LƯU KẾT QUẢ
    output_html = "results/baxat_final_decision_map.html"
    if not os.path.exists(os.path.dirname(output_html)): os.makedirs(os.path.dirname(output_html))
    m.save(output_html)
    
    print(f"✅ THÀNH CÔNG! Hãy mở file HTML để xem kết quả: {output_html}")

if __name__ == "__main__":
    visualize_baxat_decision_map()