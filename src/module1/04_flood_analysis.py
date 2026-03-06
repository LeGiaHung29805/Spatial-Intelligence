import geopandas as gpd
import pandas as pd

def analyze_risk(input_path, output_path, flood_level):
    print(f"🌊 ĐANG PHÂN TÍCH KỊCH BẢN LŨ MỨC: {flood_level}m")
    
    # Đọc dữ liệu
    buildings = gpd.read_file(input_path)

    # 1. Tính toán Độ sâu ngập lụt (Flood depth)
    # Nếu z < mực nước lũ -> ngập = (mực nước lũ - z). Nếu không -> ngập = 0
    buildings['flood_depth'] = buildings['elevation_z'].apply(lambda z: max(0, flood_level - z))

    # 2. Phân loại rủi ro chi tiết hơn dựa trên độ sâu ngập
    def classify(row):
        depth = row['flood_depth']
        z = row['elevation_z']
        
        if depth > 2: 
            return "Nguy cơ Rất cao (Ngập > 2m)"
        if depth > 0: 
            return "Nguy cơ Cao (Ngập < 2m)"
        
        # Những nhà cao hơn mực nước lũ nhưng chỉ trong khoảng 5m (vùng đệm nguy hiểm)
        if depth == 0 and (z - flood_level) <= 5: 
            return "Nguy cơ Vừa (Sát mép lũ)"
            
        return "An toàn"

    # Lưu ý: Khi dùng hàm này, bạn cần apply trên toàn bộ row thay vì chỉ cột depth
    buildings['risk_status'] = buildings.apply(classify, axis=1)

    # 3. Thống kê số lượng
    summary = buildings['risk_status'].value_counts()
    print("📊 THỐNG KÊ RỦI RO CHO BÁT XÁT:")
    print(summary)

    # 4. Tính tổng diện tích mái nhà bị ảnh hưởng
    # Lọc các nhà có độ sâu ngập > 0
    at_risk_buildings = buildings[buildings['flood_depth'] > 0]
    if 'area_in_meters' in buildings.columns:
        at_risk_area = at_risk_buildings['area_in_meters'].sum()
        print(f"🏠 Tổng diện tích mái nhà bị ngập: {at_risk_area:,.2f} m²")

    # 5. Lưu file để hiển thị Web GIS
    buildings.to_file(output_path, driver='GeoJSON')
    print(f"✅ Đã xuất file: {output_path}\n")

if __name__ == "__main__":
    input_file = "data/processed/buildings_with_z.geojson"
    
    # Bạn có thể chạy vòng lặp cho nhiều kịch bản ngập khác nhau
    scenarios = [85, 90, 95] # Các mức nước lũ giả định
    
    for level in scenarios:
        output_file = f"data/processed/flood_risk_map_{level}m.geojson"
        analyze_risk(input_file, output_file, level)