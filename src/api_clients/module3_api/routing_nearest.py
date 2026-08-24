import osmnx as ox
import networkx as nx
import numpy as np
import logging
import os
import sys
import subprocess
from dataclasses import dataclass
from threading import Lock

from pathlib import Path
from scipy.spatial import KDTree
from sqlalchemy import text
import uvicorn
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, status
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
    from src.api_clients.internal_auth import has_valid_internal_token
    from src.utils.model_validation import validate_model_files
except ImportError:
    logger.error("Không tìm thấy db_config. Hãy kiểm tra lại PYTHONPATH hoặc cấu trúc thư mục.")

# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="Bat Xat DSS AI API", version="1.0.0")


class ModelValidationRequest(BaseModel):
    modelPath: str
    scalerPath: str | None = None
    modelTarget: str

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv("AI_CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/api/v1/health')
def health_check():
    return {"status": "ok"}


@app.post('/api/v1/ai/models/validate')
def validate_model(request: ModelValidationRequest):
    try:
        return validate_model_files(
            request.modelTarget,
            request.modelPath,
            request.scalerPath,
        )
    except Exception as exc:
        logger.warning("Model validation failed: %s", exc)
        return JSONResponse(
            status_code=422,
            content={"valid": False, "message": str(exc)},
        )


def rebuild_routing_graph():
    global ROUTING_STATE
    try:
        result = subprocess.run(
            [sys.executable, str(REBUILD_SCRIPT_PATH)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            logger.error("Không thể rebuild routing graph: %s", result.stderr)
            return
        ROUTING_STATE = load_routing_graph_state()
        logger.info("Đã reload routing graph sau khi cập nhật AHP.")
    except Exception as error:
        logger.exception("Lỗi rebuild routing graph: %s", error)
    finally:
        rebuild_lock.release()


@app.post('/api/v1/ai/internal/rebuild-routing', status_code=status.HTTP_202_ACCEPTED)
def request_routing_rebuild(
    background_tasks: BackgroundTasks,
    internal_api_token: str | None = Header(default=None, alias="X-Internal-Api-Token"),
):
    if not has_valid_internal_token(INTERNAL_API_TOKEN, internal_api_token):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid internal API token")
    if not rebuild_lock.acquire(blocking=False):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Routing rebuild is already running")

    background_tasks.add_task(rebuild_routing_graph)
    return {"accepted": True}

GRAPH_PATH = PROJECT_ROOT / "models" / "baxat_mcdm_final.graphml"
REBUILD_SCRIPT_PATH = PROJECT_ROOT / "src" / "module3_routing" / "08_ahp_weighting.py"
INTERNAL_API_TOKEN = os.getenv("INTERNAL_API_TOKEN", "")
rebuild_lock = Lock()

@dataclass(frozen=True)
class RoutingGraphState:
    graph: nx.MultiGraph
    node_ids: list
    kdtree: KDTree


def load_routing_graph_state() -> RoutingGraphState:
    logger.info("Đang tải đồ thị MCDM bằng OSMnx...")
    graph = ox.load_graphml(GRAPH_PATH).to_undirected()
    for u, v, key, data in graph.edges(keys=True, data=True):
        if 'cost_safety' in data:
            data['cost_safety'] = float(data['cost_safety'])
        if 'cost_speed' in data:
            data['cost_speed'] = float(data['cost_speed'])
        if 'length' in data:
            data['length'] = float(data['length'])
    node_ids = list(graph.nodes)
    node_coords = [
        [float(data.get('y', 0)), float(data.get('x', 0))]
        for _, data in graph.nodes(data=True)
    ]
    logger.info("Đã tải xong %s nodes và tạo KDTree thành công!", len(node_ids))
    return RoutingGraphState(graph, node_ids, KDTree(node_coords))


try:
    ROUTING_STATE = load_routing_graph_state()
except Exception as e:
    logger.error(f"Không thể tải đồ thị tại {GRAPH_PATH}. Lỗi: {e}")
    sys.exit(1)

def find_nearest_node(lat: float, lng: float, routing_state: RoutingGraphState):
    """Tìm ID của node gần với tọa độ GPS nhất"""
    _, index = routing_state.kdtree.query([lat, lng])
    return routing_state.node_ids[index]

def get_path_coords(path, routing_state: RoutingGraphState):
    """Trích xuất tọa độ chi tiết từ hình học cạnh (geometry) bám sát lòng đường thực tế"""
    graph = routing_state.graph
    if not path:
        return []
    route_coords = []
    
    # Trường hợp suy biến (chỉ có 1 điểm)
    if len(path) == 1:
        return [[float(graph.nodes[path[0]]['y']), float(graph.nodes[path[0]]['x'])]]
        
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        edge_data_dict = graph.get_edge_data(u, v)
        edge_data = {}
        if edge_data_dict:
            edge_data = next(iter(edge_data_dict.values()))
            
        if 'geometry' in edge_data and edge_data['geometry']:
            geom = edge_data['geometry']
            if isinstance(geom, str):
                try:
                    from shapely import wkt
                    geom = wkt.loads(geom)
                except Exception as e:
                    logger.warning(f"Không thể tải geometry WKT: {e}")
                    geom = None
            
            if geom:
                geom_coords = list(geom.coords)
                # Xác định hướng: kiểm tra xem điểm đầu hay điểm cuối của geometry gần u hơn
                u_x = float(graph.nodes[u]['x'])
                u_y = float(graph.nodes[u]['y'])
                first_pt = geom_coords[0]
                last_pt = geom_coords[-1]
                dist_first = (first_pt[0] - u_x)**2 + (first_pt[1] - u_y)**2
                dist_last = (last_pt[0] - u_x)**2 + (last_pt[1] - u_y)**2
                
                # Nếu điểm cuối gần u hơn điểm đầu -> geometry bị ngược hướng đi, cần đảo lại
                if dist_last < dist_first:
                    geom_coords.reverse()
                    
                lat_lng_coords = [[float(pt[1]), float(pt[0])] for pt in geom_coords]
                if not route_coords:
                    route_coords.extend(lat_lng_coords)
                else:
                    route_coords.extend(lat_lng_coords[1:])
            else:
                pt_u = [float(graph.nodes[u]['y']), float(graph.nodes[u]['x'])]
                pt_v = [float(graph.nodes[v]['y']), float(graph.nodes[v]['x'])]
                if not route_coords:
                    route_coords.append(pt_u)
                route_coords.append(pt_v)
        else:
            pt_u = [float(graph.nodes[u]['y']), float(graph.nodes[u]['x'])]
            pt_v = [float(graph.nodes[v]['y']), float(graph.nodes[v]['x'])]
            if not route_coords:
                route_coords.append(pt_u)
            route_coords.append(pt_v)
            
    return route_coords

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
        routing_state = ROUTING_STATE
        start_lat = req.currentLat
        start_lng = req.currentLng
        strategy = req.strategy 
        weight_attr = 'cost_speed' if strategy == 'rescue' else 'cost_safety'

        start_node = find_nearest_node(start_lat, start_lng, routing_state)
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
            end_node = find_nearest_node(shelter['lat'], shelter['lng'], routing_state)
            if start_node == end_node:
                raw_options.append({
                    "destination": shelter, "route_coordinates": [[start_lat, start_lng]], "cost_value": 0
                })
                continue

            try:
                route_cost = nx.shortest_path_length(routing_state.graph, source=start_node, target=end_node, weight=weight_attr)
                best_route_nodes = nx.shortest_path(routing_state.graph, source=start_node, target=end_node, weight=weight_attr)
                
                # Trích xuất geometry bám đường cong mềm mại
                path_coords = get_path_coords(best_route_nodes, routing_state)
                # Nối liền mạch từ vị trí GPS người dùng
                route_coordinates = [[start_lat, start_lng]] + path_coords + [[shelter['lat'], shelter['lng']]]

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
        routing_state = ROUTING_STATE
        start_node = find_nearest_node(req.startLat, req.startLng, routing_state)
        end_node = find_nearest_node(req.endLat, req.endLng, routing_state)

        if start_node == end_node:
            return {"status": "success", "message": "Bạn đang ở đích.", "route_coordinates": []}

        try:
            route_nodes = nx.shortest_path(routing_state.graph, source=start_node, target=end_node, weight='cost_safety')
            total_cost = nx.shortest_path_length(routing_state.graph, source=start_node, target=end_node, weight='cost_safety')
            
            # Trích xuất geometry bám đường cong mềm mại
            path_coords = get_path_coords(route_nodes, routing_state)
            # Nối liên tiếp từ Marker A -> Lộ trình -> Marker B
            route_coordinates = [[req.startLat, req.startLng]] + path_coords + [[req.endLat, req.endLng]]

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
        routing_state = ROUTING_STATE
        start_node = find_nearest_node(req.startLat, req.startLng, routing_state)
        end_node = find_nearest_node(req.endLat, req.endLng, routing_state)

        scenarios = {"shortest": "length", "safety": "cost_safety", "rescue": "cost_speed"}
        results = {}

        for key, weight_attr in scenarios.items():
            try:
                path = nx.shortest_path(routing_state.graph, source=start_node, target=end_node, weight=weight_attr)
                
                # Gọi helper trích xuất geometry bám đường thực tế
                path_coords = get_path_coords(path, routing_state)
                
                # Nối liên tiếp từ Marker A -> Lộ trình -> Marker B
                coords = [[req.startLat, req.startLng]] + path_coords + [[req.endLat, req.endLng]]
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
