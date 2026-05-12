"""
Flask API Server cho GuardBatXat AI Services - Real Data Version
Endpoint: http://localhost:5000/api/v1/ai/*
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import json
import sys
import os
import warnings
import networkx as nx
import osmnx as ox
from sqlalchemy import text, create_engine

# Thêm src vào Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# ==========================================
# DATABASE & GRAPH SETUP
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GRAPH_PATH = os.path.join(PROJECT_ROOT, "models", "baxat_mcdm_final.graphml")

# Database connection (từ db_config.py)
DB_URL = "postgresql://postgres:tung2005@localhost:5432/guardbatxat"

def get_db_engine():
    """Khởi tạo engine kết nối database"""
    try:
        engine = create_engine(DB_URL, echo=False)
        engine.connect()
        logger.info("✓ Kết nối database thành công")
        return engine
    except Exception as e:
        logger.error(f"✗ Không kết nối được database: {e}")
        return None

def load_graph():
    """Nạp graph từ file GraphML"""
    try:
        if not os.path.exists(GRAPH_PATH):
            logger.warning(f"Không tìm thấy graph tại {GRAPH_PATH}")
            return None
        G = ox.load_graphml(GRAPH_PATH)
        logger.info("✓ Nạp graph thành công")
        return G
    except Exception as e:
        logger.error(f"✗ Lỗi nạp graph: {e}")
        return None

# Khởi tạo khi server start
db_engine = get_db_engine()
graph = load_graph()

# ==========================================
# HEALTH CHECK
# ==========================================
@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Kiểm tra API hoạt động"""
    return jsonify({
        "status": "success",
        "message": "AI Services đang hoạt động"
    })

# ==========================================
# ROUTING APIs
# ==========================================
@app.route('/api/v1/ai/find-safe-shelter', methods=['POST'])
def find_safe_shelter():
    """
    Tìm 3 điểm sơ tán an toàn gần nhất
    Request: {
        "currentLat": float,
        "currentLng": float,
        "strategy": "safety" or "speed"
    }
    """
    try:
        if not db_engine:
            return jsonify({
                "status": "error",
                "code": 500,
                "message": "Database không kết nối"
            }), 500
            
        data = request.get_json()
        current_lat = data.get('currentLat')
        current_lng = data.get('currentLng')
        
        logger.info(f"Tìm sơ tán gần tọa độ [{current_lat}, {current_lng}]")
        
        # Query 3 điểm sơ tán gần nhất từ database
        query = text("""
            SELECT 
                id, 
                name,
                haven_type,
                max_capacity,
                current_occupancy,
                ST_Y(geom) as latitude,
                ST_X(geom) as longitude,
                ST_DistanceSphere(geom, ST_GeomFromText('POINT(:lng :lat)', 4326)) as distance_m
            FROM batxat_safe_havens
            WHERE geom IS NOT NULL
            ORDER BY distance_m ASC
            LIMIT 3
        """)
        
        with db_engine.connect() as conn:
            result = conn.execute(query, {"lat": current_lat, "lng": current_lng})
            shelters = result.fetchall()
        
        if not shelters:
            return jsonify({
                "status": "error",
                "code": 404,
                "message": "Không tìm thấy điểm sơ tán nào"
            }), 404
        
        # Format response
        options = []
        for shelter in shelters:
            options.append({
                "shelterName": shelter[1],  # name
                "shelterLat": float(shelter[5]),  # latitude
                "shelterLng": float(shelter[6]),  # longitude
                "distance": int(shelter[7]),  # distance_m
                "estimatedTime": max(5, int(shelter[7] / 100)),  # rough estimate in minutes
                "capacity": shelter[3],  # max_capacity
                "currentOccupancy": shelter[4],  # current_occupancy
                "safetyScore": 0.85 + (0.15 * (1 - shelter[4] / max(shelter[3], 1)))  # Score based on occupancy
            })
        
        return jsonify({
            "status": "success",
            "code": 200,
            "data": {
                "options": options,
                "message": f"Tìm thấy {len(options)} điểm sơ tán an toàn"
            }
        })
        
    except Exception as e:
        logger.error(f"Lỗi tìm sơ tán: {str(e)}")
        return jsonify({
            "status": "error",
            "code": 500,
            "message": f"Lỗi hệ thống: {str(e)}"
        }), 500

@app.route('/api/v1/ai/safe-routing', methods=['POST'])
def safe_routing():
    """
    Tìm đường đi an toàn từ điểm A đến B dùng graph thực
    Request: {
        "startLat": float,
        "startLng": float,
        "endLat": float,
        "endLng": float,
        "strategyName": "cost_safety" or "cost_speed"
    }
    """
    try:
        if not graph:
            return jsonify({
                "status": "error",
                "code": 500,
                "message": "Graph mạng lưới không được nạp. Hãy chạy Bước 08 trước."
            }), 500
            
        data = request.get_json()
        start_lat = data.get('startLat')
        start_lng = data.get('startLng')
        end_lat = data.get('endLat')
        end_lng = data.get('endLng')
        strategy = data.get('strategyName') or 'cost_safety'
        
        logger.info(f"Tìm đường từ [{start_lat}, {start_lng}] đến [{end_lat}, {end_lng}] - {strategy}")
        
        # Chuẩn hóa trọng số
        G = graph.copy()
        for u, v, k, data_edge in G.edges(keys=True, data=True):
            for attr in ['cost_safety', 'cost_speed', 'length_m', 'avg_slope']:
                if attr in data_edge:
                    try:
                        data_edge[attr] = float(data_edge[attr])
                    except (ValueError, TypeError):
                        data_edge[attr] = 0.0
        
        # Tìm node gần nhất
        orig_node = ox.distance.nearest_nodes(G, X=start_lng, Y=start_lat)
        dest_node = ox.distance.nearest_nodes(G, X=end_lng, Y=end_lat)
        
        logger.info(f"Từ node {orig_node} đến node {dest_node}")
        
        # Tìm đường dùng chiến lược
        try:
            route = nx.shortest_path(G, orig_node, dest_node, weight=strategy)
        except nx.NetworkXNoPath:
            logger.warning("Không tìm thấy đường đi kết nối")
            return jsonify({
                "status": "error",
                "code": 404,
                "message": "Không tìm thấy đường đi giữa 2 điểm (mạng lưới bị đứt gãy)"
            }), 404
        
        # Trích xuất tọa độ
        route_coords = []
        total_length = 0
        total_cost = 0
        
        for i in range(len(route) - 1):
            u = route[i]
            v = route[i+1]
            edge_data = G.get_edge_data(u, v)[0]
            
            if 'geometry' in edge_data:
                try:
                    geom = edge_data['geometry']
                    if isinstance(geom, str):
                        from shapely import wkt
                        geom = wkt.loads(geom)
                    coords = list(geom.coords)
                    route_coords.extend([[x, y] for y, x in coords])
                except:
                    node_u, node_v = G.nodes[u], G.nodes[v]
                    route_coords.extend([[node_u['y'], node_u['x']], [node_v['y'], node_v['x']]])
            else:
                node_u, node_v = G.nodes[u], G.nodes[v]
                route_coords.extend([[node_u['y'], node_u['x']], [node_v['y'], node_v['x']]])
            
            total_length += float(edge_data.get('length_m', 0))
            total_cost += float(edge_data.get(strategy, 0))
        
        # Tính thời gian ước tính (5 km/h cho an toàn, 15 km/h cho tốc độ)
        avg_speed = 5 if 'safety' in strategy else 15
        estimated_time = total_length / (avg_speed * 1000 / 60)  # Convert to minutes
        
        return jsonify({
            "status": "success",
            "route_coordinates": route_coords,
            "total_mcdm_cost": round(total_cost, 2),
            "strategy": strategy,
            "distance": int(total_length),
            "estimatedTime": int(estimated_time),
            "message": f"Tìm đường thành công ({len(route)} điểm)"
        })
        
    except Exception as e:
        logger.error(f"Lỗi tìm đường: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "code": 500,
            "message": f"Lỗi hệ thống: {str(e)}"
        }), 500

@app.route('/api/v1/ai/admin-routing', methods=['POST'])
def admin_routing():
    """
    So sánh 3 lộ trình cho admin (shortest, safest, balanced)
    """
    try:
        data = request.get_json()
        
        logger.info("Admin so sánh 3 lộ trình")
        
        return jsonify({
            "status": "success",
            "data": {
                "shortest": [[10.0, 21.5], [10.001, 21.501], [10.002, 21.502]],
                "safety": [[10.0, 21.5], [10.0005, 21.5005], [10.001, 21.501], [10.002, 21.502]],
                "balanced": [[10.0, 21.5], [10.0008, 21.5008], [10.002, 21.502]]
            }
        })
        
    except Exception as e:
        logger.error(f"Lỗi admin routing: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Lỗi hệ thống: {str(e)}"
        }), 500

# ==========================================
# ERROR HANDLERS
# ==========================================
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "code": 404,
        "message": "Endpoint không tìm thấy"
    }), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({
        "status": "error",
        "code": 500,
        "message": "Lỗi máy chủ nội bộ"
    }), 500

if __name__ == '__main__':
    logger.info("🚀 Khởi động Flask API Server tại http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
