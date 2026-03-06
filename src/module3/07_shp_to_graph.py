import geopandas as gpd
import momepy
import osmnx as ox
import os
import pandas as pd

def convert_shp_to_graph():
    print("🛠️ ĐANG TÍCH HỢP TIÊU CHÍ C4, C5, C6 VÀO ĐỒ THỊ GIAO THÔNG...")
    
    shp_path = "data/shapefiles/infrastructure/Duong_Bat_Xat.shp"
    
    if not os.path.exists(shp_path):
        print(f"❌ Không tìm thấy file tại: {shp_path}")
        return

    # 1. Đọc và chuẩn hóa LineString
    gdf = gpd.read_file(shp_path)
    gdf = gdf.explode(index_parts=False)
    gdf = gdf[gdf.geometry.type == 'LineString']
    
    # Chuyển sang hệ mét để tính chiều dài chuẩn xác (m)
    gdf = gdf.to_crs(epsg=32648) 
    gdf['mm_len'] = gdf.geometry.length

    # 2. Tạo Đồ thị Topology bằng momepy
    G_nx = momepy.gdf_to_nx(gdf, approach='primal')
    
    # 3. Trích xuất Nodes và Edges
    nodes, edges = momepy.nx_to_gdf(G_nx)
    
    # --- TÍCH HỢP TIÊU CHÍ MỚI ---

    # C4: Độ rộng và Loại mặt đường (Road Capacity)
    # Giả sử cột loại đường là 'Loai_Duong' hoặc 'highway'
    def classify_road(row):
        road_type = str(row.get('Loai_Duong', row.get('highway', 'unclassified'))).lower()
        if any(x in road_type for x in ['quoc lo', 'tinh lo', 'primary', 'secondary']):
            return 1  # Đường lớn, ưu tiên cứu hộ
        return 3      # Đường nhỏ, dân sinh

    edges['road_capacity'] = edges.apply(classify_road, axis=1)

    # C5: Điểm yếu cốt tử - Cầu và Cống (Bridge Criticality)
    # Nhận diện dựa trên tên đường hoặc thuộc tính có chữ 'Cau' hoặc 'Bridge'
    def detect_bridge(row):
        text = str(row.values).lower()
        return 1 if 'cau' in text or 'bridge' in text else 0

    edges['is_bridge'] = edges.apply(detect_bridge, axis=1)

    # C6: Phản hồi cộng đồng (Crowdsourcing)
    # Khởi tạo giá trị mặc định là 0. Trường này sẽ được cập nhật từ Web/App thực tế.
    # Trong báo cáo, đây là trường dữ liệu "động" để nhận báo cáo từ người dân.
    edges['community_report'] = 0 

    # --- CHUẨN HÓA CHO OSMNX ---
    nodes = nodes.to_crs(epsg=4326)
    edges = edges.to_crs(epsg=4326)

    nodes = nodes.reset_index(drop=True)
    nodes.index.name = 'osmid'
    nodes['x'] = nodes.geometry.x
    nodes['y'] = nodes.geometry.y
    
    edges = edges.rename(columns={'node_start': 'u', 'node_end': 'v'})
    edges = edges.drop_duplicates(subset=['u', 'v'])
    edges['key'] = 0

    # Chỉ giữ lại các trường cần thiết để Module 3 sử dụng sạch sẽ
    keep_cols = ['geometry', 'mm_len', 'u', 'v', 'key', 'road_capacity', 'is_bridge', 'community_report']
    edges = edges[edges.columns.intersection(keep_cols)]
    
    edges = edges.set_index(['u', 'v', 'key'])

    try:
        G_final = ox.graph_from_gdfs(nodes, edges)
        save_path = "models/baxat_local_graph.graphml"
        ox.save_graphml(G_final, save_path)
        
        print(f"🚀 THÀNH CÔNG: Đã nhúng C4 (Hạng đường), C5 (Cầu), C6 (Cộng đồng) vào đồ thị.")
        return G_final
    except Exception as e:
        print(f"❌ Lỗi hệ thống: {e}")

if __name__ == "__main__":
    convert_shp_to_graph()