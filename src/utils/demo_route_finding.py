import os
import sys
import networkx as nx
import osmnx as ox
import folium
from shapely.geometry import LineString
import warnings

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
GRAPH_PATH = os.path.join(project_root, "models", "baxat_mcdm_final.graphml")
RESULTS_DIR = os.path.join(project_root, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

warnings.filterwarnings('ignore')

def demo_safe_routing(start_coords, end_coords, routing_strategy='cost_safety'):
    print(f"KHỞI ĐỘNG HỆ THỐNG DẪN ĐƯỜNG (Chiến lược: {routing_strategy.upper()})...")
    
    if not os.path.exists(GRAPH_PATH):
        print(f"Không tìm thấy đồ thị mạng lưới tại: {GRAPH_PATH}. Hãy chạy Bước 08 trước.")
        return

    print("🕸️ Đang nạp mạng lưới giao thông Bát Xát...")
    G = ox.load_graphml(GRAPH_PATH)
    
    print("Đang xử lý chuẩn hóa trọng số mạng lưới...")
    for u, v, k, data in G.edges(keys=True, data=True):
        for attr in ['cost_safety', 'cost_speed', 'length_m', 'avg_slope']:
            if attr in data:
                try:
                    data[attr] = float(data[attr])
                except (ValueError, TypeError):
                    data[attr] = 0.0

    print("Đang định vị tọa độ xuất phát và đích đến...")
    orig_node = ox.distance.nearest_nodes(G, X=start_coords[1], Y=start_coords[0])
    dest_node = ox.distance.nearest_nodes(G, X=end_coords[1], Y=end_coords[0])
    
    print(f"Đang chạy thuật toán tìm đường tránh rủi ro cao nhất...")
    try:
        route = nx.shortest_path(G, orig_node, dest_node, weight=routing_strategy)
    except nx.NetworkXNoPath:
        print("Không tìm thấy đường đi kết nối giữa 2 điểm này (Có thể do mạng lưới bị đứt gãy).")
        return

    route_coords = []
    total_length = 0
    total_risk_cost = 0
    max_slope_on_route = 0
    
    for i in range(len(route) - 1):
        u = route[i]
        v = route[i+1]
        edge_data = G.get_edge_data(u, v)[0] 
        
        if 'geometry' in edge_data:
            geom = edge_data['geometry']
            if isinstance(geom, str): 
                from shapely import wkt
                geom = wkt.loads(geom)
            coords = list(geom.coords)
            route_coords.extend([(y, x) for x, y in coords]) 
        else:
            node_u, node_v = G.nodes[u], G.nodes[v]
            route_coords.extend([(node_u['y'], node_u['x']), (node_v['y'], node_v['x'])])
            
        total_length += float(edge_data.get('length_m', 0))
        total_risk_cost += float(edge_data.get(routing_strategy, 0))
        slope = float(edge_data.get('avg_slope', 0))
        if slope > max_slope_on_route: max_slope_on_route = slope

    print("Đang xuất bản đồ tương tác...")
    m = folium.Map(location=[(start_coords[0] + end_coords[0])/2, 
                             (start_coords[1] + end_coords[1])/2], 
                   zoom_start=11, tiles='CartoDB dark_matter')
    
    folium.Marker(start_coords, popup="A: Vị trí Cứu hộ", icon=folium.Icon(color='blue', icon='info-sign')).add_to(m)
    folium.Marker(end_coords, popup="B: Vùng Cô lập", icon=folium.Icon(color='red', icon='flag')).add_to(m)
    
    folium.PolyLine(locations=route_coords, color="cyan", weight=6, opacity=0.8, 
                    tooltip="Tuyến đường An toàn nhất (AI Đề xuất)").add_to(m)

    dashboard_html = f'''
        <div style="position: fixed; bottom: 30px; left: 30px; width: 320px; z-index:9999; 
        background-color: rgba(20, 20, 20, 0.9); color: white; padding: 15px; border-radius: 8px; border: 1px solid cyan;">
            <h4 style="margin-top:0; color:cyan;"><b>Chỉ huy Tác chiến Giao thông</b></h4>
            <hr style="border-color: #444;">
            <small><b>Nhiệm vụ:</b> Điều hướng tránh điểm sạt lở</small><br>
            <small><b>Tổng chiều dài:</b> {total_length/1000:.2f} km</small><br>
            <small><b>Độ dốc gắt nhất trên tuyến:</b> {max_slope_on_route:.1f}°</small><br>
            <small><b>Mức độ rủi ro hệ thống:</b> {total_risk_cost:.0f} đơn vị</small><br>
            <hr style="border-color: #444;">
        </div>
    '''
    m.get_root().html.add_child(folium.Element(dashboard_html))
    
    output_path = os.path.join(RESULTS_DIR, "Demo_HoiDong_TimDuong.html")
    m.save(output_path)
    print(f"HOÀN TẤT! Hãy mở file {output_path} để trình chiếu.")

if __name__ == "__main__":
    DIEM_XUAT_PHAT = (22.540014, 103.891554) 
    DIEM_CUU_HO = (22.537064, 103.893265)   
    
    demo_safe_routing(start_coords=DIEM_XUAT_PHAT, end_coords=DIEM_CUU_HO, routing_strategy='cost_safety')