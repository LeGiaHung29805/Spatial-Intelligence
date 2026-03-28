# src/module3_routing/network_builder.py

import geopandas as gpd
import momepy
import osmnx as ox
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import numpy as np
import networkx as nx
import logging
from sqlalchemy import text
from geoalchemy2 import Geometry
from pathlib import Path
import sys

# Cấu hình Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cấu hình đường dẫn
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.CSDL.config.db_config import get_engine

SHP_PATH = PROJECT_ROOT / "data" / "shapefiles" / "infrastructure" / "Duong_Bat_Xat_New_All.shp"
SLOPE_PATH = PROJECT_ROOT / "data" / "raw" / "slope" / "Slope_BatXat_DEM_30m.tif" 
DEM_PATH = PROJECT_ROOT / "data" / "raw" / "dem" / "Copernicus_DEM_BatXat_30m.tif" 

MODELS_DIR = PROJECT_ROOT / "models"
SAVE_PATH = MODELS_DIR / "baxat_local_graph.graphml"

def build_transport_network() -> bool:
    """
    Xây dựng Topology Đồ thị Giao thông từ Shapefile.
    Đồng thời trích xuất Độ dốc và Cao độ (từ file Raster) gán vào từng đoạn đường.
    Đẩy kết quả vào PostGIS và lưu thành GraphML.
    """
    logger.info(" BƯỚC 07: TÍCH HỢP ĐỊA HÌNH VÀO MẠNG LƯỚI GIAO THÔNG...")
    
    # Kiểm tra file đầu vào
    for path in [SHP_PATH, SLOPE_PATH, DEM_PATH]:
        if not path.exists():
            logger.error(f"Không tìm thấy file: {path}")
            return False

    MODELS_DIR.mkdir(exist_ok=True)

    try:
        # 1. TIỀN XỬ LÝ VECTOR
        gdf = gpd.read_file(SHP_PATH)
        gdf = gdf.drop(columns=['u', 'v', 'key', 'osmid', 'node_start', 'node_end'], errors='ignore')
        gdf = gdf.explode(index_parts=False)
        gdf = gdf[gdf.geometry.type == 'LineString']
        
        # Ép về hệ mét (UTM Zone 48N) để tính toán topology chuẩn xác
        logger.info("Đang chuyển đổi hệ mét (EPSG:32648)...")
        gdf = gdf.to_crs(epsg=32648) 

        # 2. XÂY DỰNG TOPOLOGY BẰNG MOMEPY
        logger.info("Đang xây dựng cấu trúc Topology (Node-Edge)...")
        G_nx = momepy.gdf_to_nx(gdf, approach='primal')
        
        # Sắp xếp lại ID các Node cho liền mạch
        mapping_dict = {node: i for i, node in enumerate(G_nx.nodes())}
        G_nx = nx.relabel_nodes(G_nx, mapping_dict)
        
        nodes, edges = momepy.nx_to_gdf(G_nx)
        
        # 3. TRÍCH XUẤT RASTER VÀO VECTOR
        logger.info(f"Đang lấy mẫu địa hình trực tiếp trên {len(edges)} đoạn đường...")
        avg_slopes, avg_elevs = [], []
        
        # Vùng đệm 15m để quét raster xung quanh mặt đường
        edges_buffered = edges.geometry.buffer(15)

        with rasterio.open(SLOPE_PATH) as src_slope, rasterio.open(DEM_PATH) as src_dem:
            edges_slope_crs = edges_buffered.to_crs(src_slope.crs)
            edges_dem_crs = edges_buffered.to_crs(src_dem.crs)
            
            for i in range(len(edges)):
                s_val, d_val = 0.0, 0.0 
                try:
                    # Lấy GeoJSON-like shape để truyền vào hàm mask
                    slope_shape = [mapping(edges_slope_crs.iloc[i])]
                    dem_shape = [mapping(edges_dem_crs.iloc[i])]

                    # Trích xuất Slope
                    s_data, _ = mask(src_slope, slope_shape, crop=True)
                    valid_slopes = s_data[0][s_data[0] >= 0]
                    if valid_slopes.size > 0:
                        s_val = float(np.mean(valid_slopes))
                    
                    # Trích xuất DEM
                    d_data, _ = mask(src_dem, dem_shape, crop=True)
                    valid_dems = d_data[0][d_data[0] > -100]
                    if valid_dems.size > 0:
                        d_val = float(np.mean(valid_dems))
                        
                except ValueError as ve:
                    # Bỏ qua lỗi do polygon nằm ngoài viền bản đồ Raster
                    pass 
                except Exception as e:
                    logger.warning(f"Lỗi trích xuất tại cạnh thứ {i}: {e}")
                
                avg_slopes.append(s_val)
                avg_elevs.append(d_val)
                
                if (i+1) % 5000 == 0: 
                    logger.info(f"   ...Đã lấy mẫu {i+1} đoạn")

        # 4. CHUẨN BỊ DỮ LIỆU EDGE/NODE 
        edges['avg_slope'] = avg_slopes
        edges['avg_elevation'] = avg_elevs
        edges['length_m'] = edges.geometry.to_crs(epsg=32648).length
        edges['road_capacity'] = 3
        edges['is_bridge'] = 0
        edges['community_report'] = 0 

        # Đưa về chuẩn Web/GPS (EPSG:4326) để lưu DB
        nodes = nodes.to_crs(epsg=4326)
        edges = edges.to_crs(epsg=4326)

        nodes['x'], nodes['y'] = nodes.geometry.x, nodes.geometry.y
        nodes = nodes.reset_index().rename(columns={'index': 'node_id'})
        
        edges['u'] = edges['node_start'].astype(np.int64)
        edges['v'] = edges['node_end'].astype(np.int64)
        edges['key'] = edges.groupby(['u', 'v']).cumcount()

        # 5. GHI VÀO CSDL POSTGIS
        logger.info("Đang đẩy mạng lưới vào CSDL PostGIS...")
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS batxat_road_edges CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS batxat_road_nodes CASCADE;"))
            
        nodes[['node_id', 'x', 'y', 'geometry']].to_postgis(
            "batxat_road_nodes", engine, index=False,
            dtype={'geometry': Geometry('POINT', srid=4326)}
        )
        
        edges[['geometry', 'length_m', 'u', 'v', 'key', 'road_capacity', 'is_bridge', 'community_report', 'avg_slope', 'avg_elevation']].to_postgis(
            "batxat_road_edges", engine, index=False,
            dtype={'geometry': Geometry('LINESTRING', srid=4326)}
        )
        logger.info("Đồng bộ PostGIS thành công!")

        # 6. LƯU RA GRAPHML ĐỂ TÌM ĐƯỜNG MÀ KHÔNG CẦN CHỌC DB LIÊN TỤC
        logger.info("Đang lưu ra định dạng Đồ thị (GraphML)...")
        edges_save = edges.set_index(['u', 'v', 'key'])
        nodes_save = nodes.set_index('node_id') 
        G_final = ox.graph_from_gdfs(nodes_save, edges_save)
        ox.save_graphml(G_final, SAVE_PATH)
        
        logger.info(f"Tạo mạng lưới hoàn tất! File GraphML sẵn sàng tại: {SAVE_PATH}")
        return True

    except Exception as e:
        logger.error(f"Lỗi hệ thống trong quá trình xử lý mạng lưới: {e}")
        return False

if __name__ == "__main__":
    build_transport_network()