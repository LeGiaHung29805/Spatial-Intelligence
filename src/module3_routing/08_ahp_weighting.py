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


def get_ahp_weights(strategy: str = "safety") -> dict:
    """
    Ma trận so sánh cặp AHP 6x6 chuẩn hóa.
    Các tiêu chí: 1.Khoảng cách | 2.Ngập | 3.Sạt lở | 4.Hạng đường | 5.Cầu | 6.Cộng đồng
    """
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
    
    # Trả về dạng Dictionary giúp công thức bên dưới rõ ràng hơn
    return {
        'w_distance': float(weights[0]),
        'w_flood': float(weights[1]),
        'w_landslide': float(weights[2]),
        'w_capacity': float(weights[3]),
        'w_bridge': float(weights[4]),
        'w_report': float(weights[5])
    }


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


def apply_mcdm_to_enriched_graph() -> bool:
    logger.info("BƯỚC 08: TÍNH TOÁN CHI PHÍ MCDM TỐC ĐỘ CAO (ĐÃ TÍCH HỢP AHP)...")
    
    if not GRAPH_IN_PATH.exists(): 
        logger.error("Không tìm thấy Đồ thị gốc! Vui lòng chạy Bước 07 trước.")
        return False

    engine = get_engine()
    
    # NẠP RỦI RO REAL-TIME TỪ MODULE 2
    f_risk, l_risk = get_latest_ai_risks(engine)
    logger.info(f"Đã nhận rủi ro từ hệ thống AI: Ngập {f_risk*100}% | Sạt lở {l_risk*100}%")

    # ĐỌC ĐỒ THỊ VÀ CHUẨN BỊ PANDAS
    G = ox.load_graphml(GRAPH_IN_PATH)
    edges_list = []
    for u, v, k, data in G.edges(data=True, keys=True):
        data['u'], data['v'], data['key'] = u, v, k
        edges_list.append(data)
    
    df = pd.DataFrame(edges_list)
    numeric_cols = ['avg_slope', 'avg_elevation', 'length_m', 'road_capacity', 'is_bridge', 'community_report']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # LẤY TRỌNG SỐ AHP
    w_s = get_ahp_weights("safety")
    w_sp = get_ahp_weights("rescue")

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

    # FAST-SYNC VÀO POSTGRESQL (BULK UPDATE)
    logger.info(f"Đang Fast-Sync {len(df)} cạnh đường vào CSDL...")
    try:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE batxat_road_edges ADD COLUMN IF NOT EXISTS cost_safety NUMERIC;"))
            conn.execute(text("ALTER TABLE batxat_road_edges ADD COLUMN IF NOT EXISTS cost_speed NUMERIC;"))
            
            # Lưu trọng số AHP để Backend Spring Boot biết và hiển thị
            for strat, w in [("safety", w_s), ("rescue", w_sp)]:
                conn.execute(text("""
                    INSERT INTO batxat_ahp_weights (strategy_name, w_distance, w_flood, w_landslide, w_capacity, w_bridge, w_report)
                    VALUES (:n, :w1, :w2, :w3, :w4, :w5, :w6)
                    ON CONFLICT (strategy_name) DO UPDATE SET 
                    w_distance=EXCLUDED.w_distance, w_flood=EXCLUDED.w_flood, w_landslide=EXCLUDED.w_landslide;
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
        return False

    # CẬP NHẬT LẠI GRAPHML
    logger.info("Đang ghi đè chỉ số MCDM vào file Đồ thị...")
    cost_dict_safety = df.set_index(['u', 'v', 'key'])['cost_safety'].to_dict()
    cost_dict_speed = df.set_index(['u', 'v', 'key'])['cost_speed'].to_dict()
    
    nx.set_edge_attributes(G, cost_dict_safety, 'cost_safety')
    nx.set_edge_attributes(G, cost_dict_speed, 'cost_speed')
    
    ox.save_graphml(G, GRAPH_OUT_PATH)
    logger.info(f"HOÀN TẤT BƯỚC 08! File đồ thị MCDM cuối cùng sẵn sàng tại: {GRAPH_OUT_PATH}")
    return True

if __name__ == "__main__":
    apply_mcdm_to_enriched_graph()