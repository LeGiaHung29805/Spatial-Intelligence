import osmnx as ox
import geopandas as gpd
import os

def export_visual_pois():
    print("1. Đang chuẩn bị ranh giới Bát Xát mới...")
    boundary_path = "data/shapefiles/boundary/BatXat(moi)1.shp" 
    
    if not os.path.exists(boundary_path):
        print(f"Lỗi: Không tìm thấy ranh giới tại {boundary_path}")
        return

    # Chuẩn hóa về EPSG:4326 (Độ) để đồng bộ với OSM
    boundary_gdf = gpd.read_file(boundary_path).to_crs(epsg=4326)
    polygon = boundary_gdf.geometry.union_all()

    print("2. Định nghĩa các điểm công cộng cần tải...")
    # Mở rộng vùng quét: Quét cả Cơ sở vật chất (amenity), Cơ quan (office) và Tòa nhà (building)
    tags = {
        'amenity': ['school', 'kindergarten', 'hospital', 'clinic', 'townhall', 'police', 'fire_station', 'post_office', 'doctors', 'community_centre', 'town_hall'],
        'office': ['government'],
        'building': ['school', 'hospital', 'public', 'civic']
    }

    try:
        print("3. Đang tải dữ liệu điểm từ OpenStreetMap...")
        pois = ox.features_from_polygon(polygon, tags)
        
        if pois.empty:
            print("Không tìm thấy dữ liệu nào!")
            return

        print("4. Đang làm sạch và bảo tồn dữ liệu...")
        pois_clean = pois.copy()
        
        # Điền "Chưa cập nhật tên" vào các điểm trống thay vì xóa đi
        if 'name' not in pois_clean.columns:
            pois_clean['name'] = "Chưa cập nhật tên"
        else:
            pois_clean['name'] = pois_clean['name'].fillna("Chưa cập nhật tên")

        # Gom các tag (amenity, office, building) thành 1 cột "poi_type" duy nhất cho gọn
        for col in ['amenity', 'office', 'building']:
            if col not in pois_clean.columns:
                pois_clean[col] = ''
                
        pois_clean['poi_type'] = pois_clean['amenity'].fillna('') + pois_clean['office'].fillna('') + pois_clean['building'].fillna('')
        
        # Chỉ giữ lại 3 cột quan trọng nhất
        pois_clean = pois_clean[['name', 'poi_type', 'geometry']]

        print("5. Đồng nhất dữ liệu về định dạng Điểm (Centroid)...")
        # Đảm bảo 100% dữ liệu hình học là Điểm (Point)
        pois_clean['geometry'] = pois_clean['geometry'].centroid

        print("6. Đang tiến hành xuất file dữ liệu đôi...")
        shp_dir = "data/shapefiles/infrastructure"
        json_dir = "data/processed"
        for d in [shp_dir, json_dir]:
            if not os.path.exists(d): 
                os.makedirs(d)

        # Đảm bảo dữ liệu là chuỗi (string) để không bị lỗi khi xuất file
        pois_clean['name'] = pois_clean['name'].astype(str)
        pois_clean['poi_type'] = pois_clean['poi_type'].astype(str)

        # -- THAO TÁC 1: XUẤT FILE .GEOJSON --
        output_json = os.path.join(json_dir, "OSM_POIs_Points.geojson")
        pois_clean.to_file(output_json, driver='GeoJSON')
        print(f"   -> Đã xuất GeoJSON: {output_json}")

        # -- THAO TÁC 2: XUẤT FILE SHAPEFILE (.SHP) --
        output_shp = os.path.join(shp_dir, "OSM_POIs_Visual.shp")
        # Chuyển sang UTM Zone 48N (hệ mét) và XUẤT FILE
        pois_clean.to_file(output_shp)
        
        print(f"   -> Đã xuất Shapefile (có file .prj): {output_shp}")
        print("-" * 50)
        print(f"THÀNH CÔNG! Đã tải và xuất {len(pois_clean)} điểm dữ liệu.")
        print("-" * 50)

    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    export_visual_pois()