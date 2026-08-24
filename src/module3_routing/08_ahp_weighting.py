# src/module3_routing/ahp_mcdm_calculator.py

import osmnx as ox
import pandas as pd
import numpy as np
import networkx as nx
import logging
from sqlalchemy import text
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

# Cấu hình Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cấu hình đường dẫn hệ thống
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Import cấu hình Database
from src.CSDL.config.db_config import get_engine

GRAPH_IN_PATH = PROJECT_ROOT / "models" / "baxat_local_graph.graphml" 
GRAPH_OUT_PATH = PROJECT_ROOT / "models" / "baxat_mcdm_final.graphml"

# Khi admin đổi AHP, cần tính lại trên topology đang được API phục vụ. Bản
# baxat_local_graph là đầu vào legacy của bước 07, có thể cũ hoặc thiếu các
# thuộc tính địa hình cần cho MCDM.
GRAPH_SOURCE_PATHS = (GRAPH_OUT_PATH, GRAPH_IN_PATH)
REQUIRED_EDGE_COLUMNS = {
    'avg_slope', 'avg_elevation', 'length_m',
    'road_capacity', 'is_bridge', 'community_report',
}


WEIGHT_COLUMNS = (
    'w_distance', 'w_flood', 'w_landslide',
    'w_capacity', 'w_bridge', 'w_report'
)


def get_default_ahp_weights(strategy: str) -> dict:
    """Trọng số khởi tạo khi cơ sở dữ liệu chưa có cấu hình cho kịch bản."""
    if strategy == "safety":
        matrix = np.array([
            [1,     1/7,    1/9,    1/5,    1/6,    1/5],
            [7,     1,      1/2,    3,      2,      2  ],
            [9,     2,      1,      5,      3,      3  ],
            [5,     1/3,    1/5,    1,      1/2,    1/2],
            [6,     1/2,    1/3,    2,      1,      1  ],
            [5,     1/2,    1/3,    2,      1,      1  ] 
        ])
    else: # Kịch bản Rescue (Cứu hộ ưu tiên thời gian)
        matrix = np.array([
            [1,     2,      1/3,    1/4,    1/2,    1  ],
            [1/2,   1,      1/4,    1/5,    1/3,    1/2],
            [3,     4,      1,      1,      2,      2  ],
            [4,     5,      1,      1,      2,      3  ],
            [2,     3,      1/2,    1/2,    1,      1  ],
            [1,     2,      1/2,    1/3,    1,      1  ]
        ])
        
    weights = np.mean(matrix / np.sum(matrix, axis=0), axis=1)
    
    return dict(zip(WEIGHT_COLUMNS, map(float, weights)))


def get_ahp_weights(engine, strategy: str) -> dict:
    """Lấy trọng số do admin cấu hình; chỉ dùng ma trận mặc định để bootstrap."""
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT w_distance, w_flood, w_landslide,
                       w_capacity, w_bridge, w_report
                FROM batxat_ahp_weights
                WHERE strategy_name = :strategy
            """), {"strategy": strategy}).mappings().first()

        if row is not None:
            weights = {column: float(row[column]) for column in WEIGHT_COLUMNS}
            if all(0 <= value <= 1 for value in weights.values()) and abs(sum(weights.values()) - 1) <= 0.001:
                logger.info("Dùng trọng số AHP do admin cấu hình cho kịch bản %s", strategy)
                return weights
            logger.warning("Trọng số AHP trong DB không hợp lệ cho %s; dùng giá trị bootstrap", strategy)
    except Exception as error:
        logger.warning("Không đọc được trọng số AHP từ DB cho %s: %s", strategy, error)

    return get_default_ahp_weights(strategy)


def get_latest_ai_risks(engine) -> tuple:
    """Truy xuất % rủi ro nóng hổi nhất mà AI vừa tính ở Module 2 từ PostgreSQL"""
    query = text("""
        SELECT flood_risk_pct, landslide_risk_pct 
        FROM batxat_daily_risk 
        ORDER BY forecast_date DESC LIMIT 1
    """)
    try:
        with engine.connect() as conn:
            res = conn.execute(query).fetchone()
            if res:
                return float(res.flood_risk_pct) / 100, float(res.landslide_risk_pct) / 100
    except Exception as e:
        logger.error(f"Lỗi lấy rủi ro từ DB: {e}")
    return 0.5, 0.5 # Mặc định lấy 50% nếu hệ thống Database gặp sự cố


def load_compatible_graph():
    """Nạp đồ thị đủ dữ liệu MCDM, ưu tiên graph hiện đang được phục vụ."""
    for graph_path in GRAPH_SOURCE_PATHS:
        if not graph_path.exists():
            logger.warning("Không tìm thấy đồ thị nguồn: %s", graph_path)
            continue

        graph = ox.load_graphml(graph_path)
        available_columns = {
            column
            for _, _, _, edge_data in graph.edges(data=True, keys=True)
            for column in edge_data
        }
        missing_columns = REQUIRED_EDGE_COLUMNS - available_columns
        if missing_columns:
            logger.warning(
                "Bỏ qua đồ thị %s vì thiếu thuộc tính MCDM: %s",
                graph_path,
                ", ".join(sorted(missing_columns)),
            )
            continue

        logger.info("Dùng đồ thị MCDM nguồn: %s", graph_path)
        return graph

    logger.error("Không có đồ thị nguồn nào đủ thuộc tính cho MCDM.")
    return None


def apply_mcdm_to_enriched_graph() -> bool:
    logger.info("BƯỚC 08: TÍNH TOÁN CHI PHÍ MCDM TỐC ĐỘ CAO (ĐÃ TÍCH HỢP AHP)...")

    engine = get_engine()
    
    # NẠP RỦI RO REAL-TIME TỪ MODULE 2
    f_risk, l_risk = get_latest_ai_risks(engine)
    logger.info(f"Đã nhận rủi ro từ hệ thống AI: Ngập {f_risk*100}% | Sạt lở {l_risk*100}%")

    # ĐỌC ĐỒ THỊ VÀ CHUẨN BỊ PANDAS
    G = load_compatible_graph()
    if G is None:
        return False

    edges_list = []
    for u, v, k, data in G.edges(data=True, keys=True):
        data['u'], data['v'], data['key'] = u, v, k
        edges_list.append(data)
    
    df = pd.DataFrame(edges_list)
    numeric_cols = ['avg_slope', 'avg_elevation', 'length_m', 'road_capacity', 'is_bridge', 'community_report']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # LẤY TRỌNG SỐ AHP
    w_s = get_ahp_weights(engine, "safety")
    w_sp = get_ahp_weights(engine, "rescue")

    # TÍNH TOÁN CHI PHÍ TỔNG HỢP (Cost Matrix)
    logger.info("Đang áp dụng công thức chi phí đa tiêu chí (MCDM)...")
    local_f = (f_risk * (1000 / (df['avg_elevation'] + 1))) * 10000
    local_l = (l_risk * (df['avg_slope'] / 35.0)) * 15000

    def calc_cost(w):
        return (w['w_distance'] * df['length_m'] + 
                w['w_flood'] * local_f + 
                w['w_landslide'] * local_l + 
                w['w_capacity'] * df['road_capacity'] * 1000 + 
                w['w_bridge'] * df['is_bridge'] * 5000 + 
                w['w_report'] * df['community_report'] * 20000)

    df['cost_safety'] = calc_cost(w_s)
    df['cost_speed'] = calc_cost(w_sp)

    # Ghi graph mới vào file tạm trước. Nếu bước này lỗi, DB vẫn giữ chi phí
    # cũ đang đồng bộ với graph đang được API phục vụ.
    cost_dict_safety = df.set_index(['u', 'v', 'key'])['cost_safety'].to_dict()
    cost_dict_speed = df.set_index(['u', 'v', 'key'])['cost_speed'].to_dict()
    nx.set_edge_attributes(G, cost_dict_safety, 'cost_safety')
    nx.set_edge_attributes(G, cost_dict_speed, 'cost_speed')

    temporary_graph_path = GRAPH_OUT_PATH.with_name(
        f"{GRAPH_OUT_PATH.stem}.rebuild.graphml"
    )
    try:
        if temporary_graph_path.exists():
            temporary_graph_path.unlink()
        ox.save_graphml(G, temporary_graph_path)
    except Exception as error:
        logger.error("Không thể lưu đồ thị MCDM mới: %s", error)
        if temporary_graph_path.exists():
            temporary_graph_path.unlink()
        return False

    # FAST-SYNC VÀO POSTGRESQL (BULK UPDATE)
    logger.info(f"Đang Fast-Sync {len(df)} cạnh đường vào CSDL...")
    try:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE batxat_road_edges ADD COLUMN IF NOT EXISTS cost_safety NUMERIC;"))
            conn.execute(text("ALTER TABLE batxat_road_edges ADD COLUMN IF NOT EXISTS cost_speed NUMERIC;"))
            
            # Chỉ khởi tạo cấu hình mặc định khi DB chưa có; không ghi đè thay đổi của admin.
            for strat, w in [("safety", w_s), ("rescue", w_sp)]:
                conn.execute(text("""
                    INSERT INTO batxat_ahp_weights (strategy_name, w_distance, w_flood, w_landslide, w_capacity, w_bridge, w_report)
                    VALUES (:n, :w1, :w2, :w3, :w4, :w5, :w6)
                    ON CONFLICT (strategy_name) DO NOTHING;
                """), {"n": strat, "w1": w['w_distance'], "w2": w['w_flood'], "w3": w['w_landslide'], 
                       "w4": w['w_capacity'], "w5": w['w_bridge'], "w6": w['w_report']})

        # Đẩy vào bảng tạm
        df_sync = df[['u', 'v', 'key', 'cost_safety', 'cost_speed']]
        df_sync.to_sql('tmp_mcdm_costs', engine, if_exists='replace', index=False)
        
        # Merge và Xóa bảng tạm
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE batxat_road_edges e
                SET cost_safety = t.cost_safety,
                    cost_speed = t.cost_speed
                FROM tmp_mcdm_costs t
                WHERE e.u = t.u::bigint AND e.v = t.v::bigint AND e.key = t.key::integer;
            """))
            conn.execute(text("DROP TABLE IF EXISTS tmp_mcdm_costs;")) # Cleanup
            
        logger.info("Đồng bộ CSDL thành công rực rỡ!")
        
    except Exception as e:
        logger.error(f"Lỗi đồng bộ CSDL: {e}")
        # Cleanup dự phòng nếu có lỗi xảy ra
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS tmp_mcdm_costs;"))
        if temporary_graph_path.exists():
            temporary_graph_path.unlink()
        return False

    # Chỉ thay thế graph đang phục vụ sau khi graph mới và dữ liệu DB đều ổn.
    try:
        temporary_graph_path.replace(GRAPH_OUT_PATH)
    except Exception as error:
        logger.error("Không thể kích hoạt đồ thị MCDM mới: %s", error)
        if temporary_graph_path.exists():
            temporary_graph_path.unlink()
        return False

    logger.info(f"HOÀN TẤT BƯỚC 08! File đồ thị MCDM cuối cùng sẵn sàng tại: {GRAPH_OUT_PATH}")
    return True

if __name__ == "__main__":
    apply_mcdm_to_enriched_graph()
