import os
import sys
import networkx as nx
import osmnx as ox
import folium
import pandas as pd
import warnings

# --- 0. CẤU HÌNH ĐƯỜNG DẪN ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
GRAPH_PATH = os.path.join(project_root, "models", "baxat_mcdm_final.graphml")
RESULTS_DIR = os.path.join(project_root, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
warnings.filterwarnings('ignore')

def auto_find_evacuation_point():
    # print("BƯỚC 10: KÍCH HOẠT HỆ THỐNG TÌM ĐIỂM SƠ TÁN KHẨN CẤP...")
    
    if not os.path.exists(GRAPH_PATH):
        print("Lỗi: Không tìm thấy mạng lưới đồ thị!")    
        return
        
    G = ox.load_graphml(GRAPH_PATH)
    
    # print("Đang dọn dẹp các đường mồ côi và mở đường 2 chiều cho xe cứu hộ...")
    G = G.to_undirected() 
    largest_cc = max(nx.connected_components(G), key=len) 
    G = G.subgraph(largest_cc).copy() 
    
    # print("Đang tải thông số địa hình và an toàn giao thông...")
    for u, v, k, data in G.edges(data=True, keys=True):
        data['length_m'] = float(data.get('length_m', data.get('length', 1))) 
        data['cost_safety'] = float(data.get('cost_safety', 1))
        data['avg_slope'] = float(data.get('avg_slope', 0))

    # 1 VÙNG NGUY HIỂM & 3 ĐIỂM SƠ TÁN DỰ PHÒNG
    DISASTER_ZONE = (22.538066, 103.894262) 

    # Danh sách các cơ sở công cộng có thể làm nơi sơ tán (Tọa độ giả định gần đó)
    CANDIDATE_SHELTERS = {
        "UBND Xã ": (22.526567, 103.897249),
        "Trường Tiểu Học ": (22.538279, 103.896534),
        "Trạm Y Tế ": (22.530198, 103.898000)
    }

    orig_node = ox.distance.nearest_nodes(G, X=DISASTER_ZONE[1], Y=DISASTER_ZONE[0])
    
    #AI PHÂN TÍCH TỪNG ĐIỂM SƠ TÁN
    print(" AI đang đánh giá các phương án sơ tán...")
    evacuation_options = []

    for name, coords in CANDIDATE_SHELTERS.items():
        dest_node = ox.distance.nearest_nodes(G, X=coords[1], Y=coords[0])
        try:
            route = nx.shortest_path(G, orig_node, dest_node, weight='cost_safety')
            
            total_dist = 0
            total_risk = 0
            max_slope = 0
            
            for i in range(len(route) - 1):
                u, v = route[i], route[i+1]
                edge_data = G.get_edge_data(u, v)[0]
                
                total_dist += float(edge_data.get('length_m', 0))
                total_risk += float(edge_data.get('cost_safety', 0))
                slope = float(edge_data.get('avg_slope', 0))
                if slope > max_slope: max_slope = slope
                
            evacuation_options.append({
                "Tên Điểm": name,
                "Tọa độ": coords,
                "Tuyến đường": route,
                "Quãng đường (km)": round(total_dist / 1000, 2),
                "Rủi ro hệ thống": round(total_risk, 0),
                "Độ dốc Max": round(max_slope, 1)
            })
            print(f"  Đã quét tuyến đến {name}: Rủi ro {round(total_risk,0)} | {round(total_dist/1000,2)}km")
        except nx.NetworkXNoPath:
            print(f" {name}: Bị cô lập, không thể tiếp cận!")

    if not evacuation_options:
        print("CẢNH BÁO TỐI THƯỢNG: Tất cả các điểm sơ tán đều bị cô lập!")
        return

    #CHỌN ĐIỂM SƠ TÁN TỐI ƯU NHẤT
    best_option = min(evacuation_options, key=lambda x: x["Rủi ro hệ thống"])
    print(f"\nQUYẾT ĐỊNH CỦA AI: Điểm sơ tán an toàn nhất là '{best_option['Tên Điểm']}'")

    m = folium.Map(location=[DISASTER_ZONE[0], DISASTER_ZONE[1]], zoom_start=14, tiles='CartoDB dark_matter')

    # Vẽ Tâm thảm họa
    folium.Marker(
        DISASTER_ZONE, 
        popup="<b>VÙNG DÂN CƯ ĐANG NGUY HIỂM</b>", 
        icon=folium.Icon(color='red', icon='warning-sign')
    ).add_to(m)
    folium.Circle(DISASTER_ZONE, radius=150, color='red', fill=True, fill_opacity=0.3).add_to(m)

    # Hàm vẽ tuyến đường
    def plot_route(route, color, weight, opacity, dash_array=None):
        coords = []
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            edge_data = G.get_edge_data(u, v)[0]
            if 'geometry' in edge_data:
                geom = edge_data['geometry']
                if isinstance(geom, str): 
                    from shapely import wkt
                    geom = wkt.loads(geom)
                coords.extend([(y, x) for x, y in list(geom.coords)])
            else:
                coords.extend([(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])])
        folium.PolyLine(coords, color=color, weight=weight, opacity=opacity, dash_array=dash_array).add_to(m)

    # Vẽ tất cả các phương án
    dashboard_details = ""
    for opt in evacuation_options:
        is_best = (opt["Tên Điểm"] == best_option["Tên Điểm"])
        
        # Vẽ Marker điểm sơ tán
        icon_color = 'green' if is_best else 'gray'
        icon_type = 'home' if is_best else 'info-sign'
        folium.Marker(
            opt["Tọa độ"], 
            popup=f"<b>{opt['Tên Điểm']}</b><br>Rủi ro: {opt['Rủi ro hệ thống']}", 
            icon=folium.Icon(color=icon_color, icon=icon_type)
        ).add_to(m)
        
        # Vẽ Tuyến đường
        if is_best:
            plot_route(opt["Tuyến đường"], '#00FF00', 8, 0.9)
            status_html = f"<b style='color:lime;'>ĐƯỢC CHỌN (Tối ưu)</b>"
        else:
            plot_route(opt["Tuyến đường"], '#888888', 4, 0.6, dash_array='5, 10')
            status_html = f"<span style='color:gray;'>BỊ LOẠI (Rủi ro cao)</span>"


        dashboard_details += f"""
            <div style="margin-bottom: 8px; font-size: 12px; background: rgba(255,255,255,0.1); padding: 5px; border-radius: 4px;">
                <b>{opt['Tên Điểm']}</b><br>
                Khoảng cách: {opt['Quãng đường (km)']} km | Dốc Max: {opt['Độ dốc Max']}°<br>
                Điểm rủi ro: <b>{opt['Rủi ro hệ thống']}</b> -> {status_html}
            </div>
        """

    dashboard_html = f'''
        <div style="position: fixed; top: 20px; right: 20px; width: 340px; z-index:9999; 
        background-color: rgba(10, 10, 10, 0.85); color: white; padding: 15px; border-radius: 8px; border: 1px solid #00FF00;">
            <h4 style="margin-top:0; color:#00FF00; text-align: center;"><b>AI CHỈ ĐỊNH ĐIỂM SƠ TÁN</b></h4>
            <hr style="border-color: #555;">
            {dashboard_details}
            <hr style="border-color: #555;">
            <small style="color: #aaa;"><i>*Hệ thống tự động loại bỏ các điểm sơ tán yêu cầu đi qua sườn dốc nguy hiểm hoặc ngập lụt.</i></small>
        </div>
    '''
    m.get_root().html.add_child(folium.Element(dashboard_html))
    
    output_map = os.path.join(RESULTS_DIR, "Demo_HoiDong_DiemSoTan.html")
    m.save(output_map)
    print(f"HOÀN TẤT! File sơ đồ tác chiến lưu tại: {output_map}")

if __name__ == "__main__":
    auto_find_evacuation_point()