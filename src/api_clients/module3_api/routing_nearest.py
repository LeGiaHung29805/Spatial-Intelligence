import osmnx as ox
import networkx as nx
import numpy as np
import logging
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS 
from scipy.spatial import KDTree
from sqlalchemy import text

# Cấu hình Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cấu hình đường dẫn hệ thống
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.CSDL.config.db_config import get_engine

app = Flask(__name__)
CORS(app)  # Cho phép mọi Domain (như localhost:3000) gọi API này

GRAPH_PATH = PROJECT_ROOT / "models" / "baxat_mcdm_final.graphml"

# KHỞI TẠO DỮ LIỆU ĐỒ THỊ (Chạy 1 lần khi Start API)
logger.info("Đang tải đồ thị MCDM bằng OSMnx...")
try:
    G = ox.load_graphml(GRAPH_PATH)
    G = G.to_undirected()
    for u, v, key, data in G.edges(keys=True, data=True):
        if 'cost_safety' in data:
            data['cost_safety'] = float(data['cost_safety'])
        if 'cost_speed' in data:
            data['cost_speed'] = float(data['cost_speed'])
except Exception as e:
    logger.error(f"Không thể tải đồ thị, hãy chắc chắn đã chạy Bước 08. Lỗi: {e}")
    sys.exit(1)

# Lấy danh sách Node và chuẩn bị tọa độ cho KDTree
node_ids = list(G.nodes)
node_coords = []

for node, data in G.nodes(data=True):
    # Trong osmnx, x là longitude, y là latitude
    lat = float(data.get('y', 0))
    lng = float(data.get('x', 0))
    node_coords.append([lat, lng])

kdtree = KDTree(node_coords)
logger.info(f"Đã tải xong {len(node_ids)} nodes và tạo KDTree thành công!")

def find_nearest_node(lat, lng):
    """Tìm ID của node gần với tọa độ GPS nhất"""
    distance, index = kdtree.query([lat, lng])
    return node_ids[index]

# API ENDPOINT TÌM ĐIỂM SƠ TÁN (PHIÊN BẢN CHỌN TOP 3 ĐỊA ĐIỂM)
@app.route('/api/v1/ai/find-safe-shelter', methods=['POST'])
def find_safe_shelter():
    try:
        data = request.json
        start_lat = float(data.get('currentLat'))
        start_lng = float(data.get('currentLng'))
        
        strategy = data.get('strategy', 'safety') 
        weight_attr = 'cost_speed' if strategy == 'rescue' else 'cost_safety'

        start_node = find_nearest_node(start_lat, start_lng)

        engine = get_engine()
        shelters = []
        
        with engine.connect() as conn:
            state_query = text("SELECT active_flood_level FROM batxat_system_state LIMIT 1")
            system_state = conn.execute(state_query).fetchone()
            current_flood_level = float(system_state.active_flood_level) if system_state else 0.0


            query = text("""
                SELECT DISTINCT ON (sh.id)
                    sh.id, sh.name, sh.latitude, sh.longitude, 
                    sh.max_capacity, sh.current_occupancy
                FROM batxat_safe_havens sh
                JOIN LATERAL simulate_flood_risk(:f_level) sim ON ST_DWithin(sh.geom, sim.geom, 0.0001)
                JOIN LATERAL get_combined_landslide_risk() ls ON ST_DWithin(sh.geom, ls.geom, 0.0001)
                WHERE sh.is_accessible = TRUE 
                  AND sh.current_occupancy < sh.max_capacity
                  AND sim.flood_depth = 0 
                  AND sim.risk_status NOT LIKE '%Nguy cơ Cao%'
                  AND sim.risk_status NOT LIKE '%Nguy cơ Rất cao%'
                  AND ls.risk_severity NOT IN ('Rất Cao (Nguy cấp)', 'Cao')
            """)
            
            result = conn.execute(query, {"f_level": current_flood_level})
            for row in result:
                shelters.append({
                    "id": row.id,
                    "name": row.name,
                    "lat": row.latitude,
                    "lng": row.longitude,
                    "max_capacity": row.max_capacity,
                    "current_occupancy": row.current_occupancy
                })

        if not shelters:
            return jsonify({"status": "fail", "message": "Không có điểm sơ tán nào còn chỗ hoặc an toàn khỏi lũ/sạt lở."}), 404

        raw_options = []

        for shelter in shelters:
            end_node = find_nearest_node(shelter['lat'], shelter['lng'])
            
            if start_node == end_node:
                raw_options.append({
                    "destination": {
                        "id": shelter['id'],
                        "name": shelter['name'],
                        "lat": shelter['lat'],
                        "lng": shelter['lng'],
                        "available_capacity": shelter['max_capacity'] - shelter['current_occupancy']
                    },
                    "route_coordinates": [[start_lat, start_lng]],
                    "cost_value": 0
                })
                continue

            try:
                route_cost = nx.shortest_path_length(G, source=start_node, target=end_node, weight=weight_attr)
                best_route_nodes = nx.shortest_path(G, source=start_node, target=end_node, weight=weight_attr)
                
                route_coordinates = []
                for node in best_route_nodes:
                    node_data = G.nodes[node]
                    route_coordinates.append([float(node_data['y']), float(node_data['x'])])

                raw_options.append({
                    "destination": {
                        "id": shelter['id'],
                        "name": shelter['name'],
                        "lat": shelter['lat'],
                        "lng": shelter['lng'],
                        "available_capacity": shelter['max_capacity'] - shelter['current_occupancy']
                    },
                    "route_coordinates": route_coordinates,
                    "cost_value": round(route_cost, 2)
                })
            except nx.NetworkXNoPath:
                continue

        # Sắp xếp danh sách dựa trên chi phí
        raw_options.sort(key=lambda x: x['cost_value'])

        # FIX 2: BỘ LỌC ĐẢM BẢO CHỈ LẤY CÁC TÒA NHÀ DUY NHẤT (ĐỀ PHÒNG TRƯỜNG HỢP CÙNG 1 NODE)
        unique_options = []
        seen_ids = set()
        for opt in raw_options:
            shelter_id = opt['destination']['id']
            if shelter_id not in seen_ids:
                seen_ids.add(shelter_id)
                unique_options.append(opt)

        top_3_options = unique_options[:3]

        if top_3_options:
            return jsonify({
                "status": "success",
                "message": f"Đã tìm thấy {len(top_3_options)} điểm sơ tán an toàn nhất xung quanh bạn.",
                "strategy_used": strategy,
                "options": top_3_options
            })
        else:
            return jsonify({"status": "fail", "message": "Các tuyến đường đều bị cô lập do thiên tai."}), 404

    except Exception as e:
        logger.error(f"Lỗi xử lý API: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
# ==========================================
# API TÌM LỘ TRÌNH AN TOÀN NHẤT (A -> B)
# ==========================================
@app.route('/api/v1/ai/safe-routing', methods=['POST'])
def find_safe_route():
    try:
        data = request.json
        start_lat = float(data.get('startLat'))
        start_lng = float(data.get('startLng'))
        end_lat = float(data.get('endLat'))
        end_lng = float(data.get('endLng'))

        # Tìm Node trên đồ thị gần nhất với tọa độ A và B
        start_node = find_nearest_node(start_lat, start_lng)
        end_node = find_nearest_node(end_lat, end_lng)

        if start_node == end_node:
            return jsonify({
                "status": "success", 
                "message": "Bạn đang ở rất gần điểm đến.", 
                "route_coordinates": []
            })

        try:
            # SỨC MẠNH CỦA AI: Dùng weight='cost_safety' để thuật toán tự động né vùng đỏ
            route_nodes = nx.shortest_path(G, source=start_node, target=end_node, weight='cost_safety')
            total_cost = nx.shortest_path_length(G, source=start_node, target=end_node, weight='cost_safety')

            # Chuyển đổi ID Node thành mảng tọa độ [Lat, Lng] cho bản đồ
            route_coordinates = []
            for node in route_nodes:
                node_data = G.nodes[node]
                route_coordinates.append([float(node_data['y']), float(node_data['x'])])

            return jsonify({
                "status": "success",
                "message": "Đã tìm thấy lộ trình an toàn nhất (đã tránh các điểm ngập/sạt lở).",
                "route_coordinates": route_coordinates,
                "total_mcdm_cost": round(total_cost, 2)
            })

        except nx.NetworkXNoPath:
            return jsonify({
                "status": "fail", 
                "message": "Không tìm thấy đường đi an toàn. Vị trí đích có thể đã bị cô lập hoàn toàn."
            }), 404

    except Exception as e:
        logger.error(f"Lỗi API safe-routing: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)