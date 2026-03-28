import geopandas as gpd
import pandas as pd
import os

def analyze_risk_integrated(input_path, output_path, flood_level):
    print(f"🌊 ĐANG PHÂN TÍCH KỊCH BẢN LŨ MỨC: {flood_level}m (Tích hợp Thủy văn)")
    
    #Đọc dữ liệu đã có dist_to_water và elevation_z
    buildings = gpd.read_file(input_path)

    #Tính toán Độ sâu ngập lụt
    buildings['flood_depth'] = buildings['elevation_z'].apply(lambda z: max(0, flood_level - z))

    # Logic phân loại rủi ro đa tiêu chí
    def classify_enhanced(row):
        depth = row['flood_depth']
        dist_water = row['dist_to_water']
        z = row['elevation_z']
        
        # ƯU TIÊN 1: Nguy cơ Rất cao (Ngập sâu)
        if depth > 2: 
            return "Nguy cơ Rất cao (Ngập > 2m)"
        
        # ƯU TIÊN 2: Nguy cơ Cao (Ngập hoặc Sát mép nước)
        if depth > 0: 
            return "Nguy cơ Cao (Bị ngập)"
        
        if dist_water < 50: # Nếu cách sông/hồ dưới 50m
            return "Nguy cơ Cao (Sát mép nước/Sạt lở bờ)"
        
        # ƯU TIÊN 3: Nguy cơ Vừa (Vùng đệm an toàn)
        # Những nhà cao hơn mực nước lũ nhưng chỉ trong khoảng 5m
        if (z - flood_level) <= 5: 
            return "Nguy cơ Vừa (Sát mép lũ)"
            
        return "An toàn"

    # Ghi đè trạng thái rủi ro cũ bằng kết quả phân tích mới
    buildings['risk_status'] = buildings.apply(classify_enhanced, axis=1)

    #Thống kê số lượng
    summary = buildings['risk_status'].value_counts()
    print("\nTHỐNG KÊ RỦI RO TỔNG HỢP:")
    print(summary)

    # Tính tổng diện tích mái nhà bị ảnh hưởng (Nguy cơ Cao và Rất cao)
    at_risk_buildings = buildings[buildings['risk_status'].str.contains("Nguy cơ")]
    if 'area_in_meters' in buildings.columns:
        at_risk_area = at_risk_buildings['area_in_meters'].sum()
        print(f"Tổng diện tích mái nhà trong vùng rủi ro: {at_risk_area:,.2f} m²")

    buildings.to_file(output_path, driver='GeoJSON')
    print(f"Đã xuất file: {output_path}\n")

if __name__ == "__main__":
    input_file = "data/processed/buildings_with_hydro.geojson"
    
    scenarios = [80, 82, 83.5] 
    
    for level in scenarios:
        output_file = f"data/processed/flood_risk_map_{level}m.geojson"
        analyze_risk_integrated(input_file, output_file, level)