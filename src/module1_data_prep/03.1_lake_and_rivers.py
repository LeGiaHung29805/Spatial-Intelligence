import geopandas as gpd
import pandas as pd
import os

def extract_hydrology_features():
    print("ĐANG TÍCH HỢP DỮ LIỆU SÔNG HỒ...")
    
    build_path = "data/processed/buildings_with_z.geojson"
    river_path = "data/shapefiles/hydrology/Song_ngoi_Bat_Xat_1.shp"
    lake_path = "data/shapefiles/hydrology/Ho_Bat_Xat_1.shp"
    output_path = "data/processed/buildings_with_hydro.geojson"

    # Kiểm tra file đầu vào
    if not os.path.exists(build_path):
        print(f"❌ Lỗi: Không tìm thấy file {build_path}. Hãy chạy Bước 03 trước!")
        return

    #Nạp dữ liệu
    buildings = gpd.read_file(build_path)
    rivers = gpd.read_file(river_path)
    lakes = gpd.read_file(lake_path)

    #Chuyển sang hệ mét (UTM 48N) để tính khoảng cách chính xác bằng mét
    buildings = buildings.to_crs(epsg=32648)
    rivers = rivers.to_crs(epsg=32648)
    lakes = lakes.to_crs(epsg=32648)

    #Gom tất cả sông và hồ thành một khối duy nhất để tính toán cho nhanh
    print(" Đang gộp dữ liệu sông ngòi và mặt nước...")
    water_geoms = pd.concat([rivers.geometry, lakes.geometry])
    union_water = water_geoms.unary_union # Gom tất cả thành 1 MultiLine/MultiPolygon

    #Tính khoảng cách từ mỗi ngôi nhà đến nguồn nước gần nhất
    print("Đang đo khoảng cách thực tế đến sông/hồ...")
    # buildings.geometry.distance() sẽ tính khoảng cách ngắn nhất đến water
    buildings['dist_to_water'] = buildings.geometry.distance(union_water)

    if 'risk_status' not in buildings.columns:
        buildings['risk_status'] = "Chưa phân loại"

    # Cập nhật trạng thái dựa trên khoảng cách thủy văn
    def update_risk(row):
        dist = row['dist_to_water']
        # Nếu nhà rất gần nước (< 50m), đánh dấu là vùng nhạy cảm
        if dist < 50:
            return "Vùng đệm sông hồ (Nguy cơ cao)"
        return row['risk_status']

    buildings['risk_status'] = buildings.apply(update_risk, axis=1)

    buildings.to_crs(epsg=4326).to_file(output_path, driver='GeoJSON')
    print(f"Hoàn thành! Đã thêm cột 'dist_to_water' vào {len(buildings)} ngôi nhà.")
    print(f"File lưu tại: {output_path}")

if __name__ == "__main__":
    extract_hydrology_features()