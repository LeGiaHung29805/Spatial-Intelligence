import osmnx as ox
import geopandas as gpd
import os

def download_roads_raw_4326():
    print("🌍 ĐANG TẢI DỮ LIỆU NGUYÊN BẢN (KHÔNG HÀN GẮN - KHÔNG BIẾN ĐỔI CRS)...")

    # 1. Tắt bộ nhớ đệm để lấy dữ liệu mới nhất
    ox.settings.use_cache = False

    # 2. Đọc ranh giới chuẩn
    boundary_shp_path = "data/shapefiles/boundary/BatXat(moi)1.shp"
    if not os.path.exists(boundary_shp_path):
        print(f"❌ Không tìm thấy ranh giới tại: {boundary_shp_path}")
        return

    # Luôn ép ranh giới về 4326 để đồng bộ với OSM
    boundary_gdf = gpd.read_file(boundary_shp_path).to_crs(epsg=4326)
    polygon = boundary_gdf.geometry.union_all() 

    # Bộ lọc lấy tất cả các loại đường
    cf = '["highway"]' 

    try:
        # 3. TẢI ĐỒ THỊ TRỰC TIẾP TỪ OSM
        # simplify=False: Giữ nguyên chi tiết từng mét đường
        # retain_all=True: Giữ lại toàn bộ các đoạn đường màu vàng rời rạc
        G = ox.graph_from_polygon(polygon, custom_filter=cf, simplify=False, retain_all=True)
        
        # Ép nhãn hệ tọa độ chuẩn cho đồ thị
        G.graph["crs"] = "EPSG:4326"
        
        # Kiểm tra tọa độ ngay lập tức
        test_node = list(G.nodes(data=True))[0][1]
        print(f"📍 Tọa độ gốc OSM: Long={test_node['x']:.4f}, Lat={test_node['y']:.4f}")

        # 4. LƯU DỮ LIỆU
        save_dir = "data/shapefiles/infrastructure"
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        # Lưu file GraphML (Đây là file gốc cho mọi bước sau)
        graphml_path = os.path.join(save_dir, "Duong_Bat_Xat_Connected.graphml")
        ox.save_graphml(G, filepath=graphml_path)
        
        # 5. XUẤT SHAPEFILE (Bản sửa lỗi Engine Pyogrio)
        _, edges = ox.graph_to_gdfs(G)
        
        # Làm sạch cột dữ liệu (Chuyển list/dict thành string)
        for col in edges.columns:
            edges[col] = edges[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)

        # Đảm bảo GeoDataFrame mang hệ tọa độ 4326 trước khi ghi file
        edges = edges.set_crs("EPSG:4326", allow_override=True)
        
        shp_output = os.path.join(save_dir, "Duong_Bat_Xat_New.shp")
        # Ghi file không truyền tham số 'crs' để tránh lỗi engine
        edges.to_file(shp_output, encoding='utf-8')

        print("-" * 50)
        print("🚀 THÀNH CÔNG: Dữ liệu đã được tải về nguyên bản 100%.")
        print(f"📂 Đã khớp hoàn toàn với ranh giới của bạn.")
        print("-" * 50)

    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    download_roads_raw_4326()