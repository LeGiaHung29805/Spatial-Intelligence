import osmnx as ox
import networkx as nx
import numpy as np
import logging
import sys

from pathlib import Path
from scipy.spatial import KDTree
from sqlalchemy import text
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ==========================================
# CẤU HÌNH LOGGING & HỆ THỐNG
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Lùi lại 3 cấp: module3_api -> api_clients -> src -> Spatial-Intelligence
PROJECT_ROOT = Path(__file__).resolve().parents[3] 
sys.path.append(str(PROJECT_ROOT))

# Đảm bảo đường dẫn file đồ thị trỏ đúng về root/models
GRAPH_PATH = PROJECT_ROOT / "models" / "baxat_mcdm_final.graphml"

# Đảm bảo import đúng sau khi bạn đã cấu trúc lại thư mục bằng Git
try:
    from src.CSDL.config.db_config import get_engine
except ImportError:
    logger.error("Không tìm thấy db_config. Hãy kiểm tra lại PYTHONPATH hoặc cấu trúc thư mục.")

# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="Bat Xat DSS AI API", version="1.0.0")

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GRAPH_PATH = PROJECT_ROOT / "models" / "baxat_mcdm_final.graphml"

# ==========================================
# KHỞI TẠO DỮ LIỆU ĐỒ THỊ
# ==========================================
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
    logger.error(f"Không thể tải đồ thị tại {GRAPH_PATH}. Lỗi: {e}")
    sys.exit(1)

node_ids = list(G.nodes)
node_coords = []

for node, data in G.nodes(data=True):
    lat = float(data.get('y', 0))
    lng = float(data.get('x', 0))
    node_coords.append([lat, lng])

kdtree = KDTree(node_coords)
logger.info(f"Đã tải xong {len(node_ids)} nodes và tạo KDTree thành công!")

def find_nearest_node(lat: float, lng: float):
    """Tìm ID của node gần với tọa độ GPS nhất"""
    distance, index = kdtree.query([lat, lng])
    return node_ids[index]

# ==========================================
# ĐỊNH NGHƠI MODEL DỮ LIỆU ĐẦU VÀO
# ==========================================
class ShelterRequest(BaseModel):
    currentLat: float
    currentLng: float
    strategy: str = 'safety'

class SafeRouteRequest(BaseModel):
    startLat: float
    startLng: float
    endLat: float
    endLng: float

class AdminRouteRequest(BaseModel):
    startLat: float
    startLng: float
    endLat: float
    endLng: float

# ==========================================
# API ENDPOINT TÌM ĐIỂM SƠ TÁN
# ==========================================
@app.post('/api/v1/ai/find-safe-shelter')
def find_safe_shelter(req: ShelterRequest):
    try:
        start_lat = req.currentLat
        start_lng = req.currentLng
        strategy = req.strategy 
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
                    "id": row.id, "name": row.name, "lat": row.latitude, "lng": row.longitude,
                    "max_capacity": row.max_capacity, "current_occupancy": row.current_occupancy
                })

        if not shelters:
            return JSONResponse(status_code=404, content={"status": "fail", "message": "Không có điểm sơ tán an toàn."})

        raw_options = []
        for shelter in shelters:
            end_node = find_nearest_node(shelter['lat'], shelter['lng'])
            if start_node == end_node:
                raw_options.append({
                    "destination": shelter, "route_coordinates": [[start_lat, start_lng]], "cost_value": 0
                })
                continue

            try:
                route_cost = nx.shortest_path_length(G, source=start_node, target=end_node, weight=weight_attr)
                best_route_nodes = nx.shortest_path(G, source=start_node, target=end_node, weight=weight_attr)
                route_coordinates = [[float(G.nodes[n]['y']), float(G.nodes[n]['x'])] for n in best_route_nodes]

                raw_options.append({
                    "destination": {
                        "id": shelter['id'], "name": shelter['name'], "lat": shelter['lat'], "lng": shelter['lng'],
                        "available_capacity": shelter['max_capacity'] - shelter['current_occupancy']
                    },
                    "route_coordinates": route_coordinates,
                    "cost_value": round(route_cost, 2)
                })
            except nx.NetworkXNoPath:
                continue

        raw_options.sort(key=lambda x: x['cost_value'])
        unique_options = []
        seen_ids = set()
        for opt in raw_options:
            if opt['destination']['id'] not in seen_ids:
                seen_ids.add(opt['destination']['id'])
                unique_options.append(opt)

        top_3 = unique_options[:3]
        return {"status": "success", "message": f"Tìm thấy {len(top_3)} điểm sơ tán.", "options": top_3}
    except Exception as e:
        logger.error(f"Lỗi find-safe-shelter: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# ==========================================
# API TÌM LỘ TRÌNH AN TOÀN NHẤT (A -> B)
# ==========================================
@app.post('/api/v1/ai/safe-routing')
def find_safe_route(req: SafeRouteRequest):
    try:
        start_node = find_nearest_node(req.startLat, req.startLng)
        end_node = find_nearest_node(req.endLat, req.endLng)

        if start_node == end_node:
            return {"status": "success", "message": "Bạn đang ở đích.", "route_coordinates": []}

        try:
            route_nodes = nx.shortest_path(G, source=start_node, target=end_node, weight='cost_safety')
            total_cost = nx.shortest_path_length(G, source=start_node, target=end_node, weight='cost_safety')
            route_coordinates = [[float(G.nodes[n]['y']), float(G.nodes[n]['x'])] for n in route_nodes]

            return {
                "status": "success",
                "route_coordinates": route_coordinates,
                "total_mcdm_cost": round(total_cost, 2)
            }
        except nx.NetworkXNoPath:
            return JSONResponse(status_code=404, content={"status": "fail", "message": "Đích đến bị cô lập."})
    except Exception as e:
        logger.error(f"Lỗi API safe-routing: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# ==========================================
# API DÀNH RIÊNG CHO ADMIN (So sánh lộ trình)
# ==========================================
@app.post('/api/v1/ai/admin-routing')
def admin_compare_routing(req: AdminRouteRequest):
    try:
        start_node = find_nearest_node(req.startLat, req.startLng)
        end_node = find_nearest_node(req.endLat, req.endLng)

        scenarios = {"shortest": "length", "safety": "cost_safety", "rescue": "cost_speed"}
        results = {}

        for key, weight_attr in scenarios.items():
            try:
                path = nx.shortest_path(G, source=start_node, target=end_node, weight=weight_attr)
                coords = [[req.startLat, req.startLng]]
                for node in path:
                    coords.append([float(G.nodes[node]['y']), float(G.nodes[node]['x'])])
                coords.append([req.endLat, req.endLng])
                results[key] = coords
            except nx.NetworkXNoPath:
                results[key] = [] 

        if not any(results.values()):
            return JSONResponse(status_code=404, content={"status": "fail", "message": "Không tìm thấy đường."})

        return {"status": "success", "data": results}
    except Exception as e:
        logger.error(f"Lỗi admin-routing: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# ==========================================
# CHẠY SERVER
# ==========================================
if __name__ == '__main__':
    # Đảm bảo tên file của bạn là routing_nearest.py, nếu không hãy đổi lại chuỗi bên dưới
    uvicorn.run("routing_nearest:app", host="0.0.0.0", port=5000, reload=True)