import geopandas as gpd
import pandas as pd
import os

def calculate_capacity_multi():
    # 1. Danh sách các kịch bản bạn đã chọn
    scenarios = [85, 90, 95]
    
    print(f"🚀 BẮT ĐẦU PHÂN TÍCH SỨC CHỨA CHO {len(scenarios)} KỊCH BẢN...")

    for level in scenarios:
        input_path = f"data/processed/flood_risk_map_{level}m.geojson"
        output_path = f"data/processed/final_analysis_{level}m.geojson"

        if not os.path.exists(input_path):
            print(f"⚠️ Bỏ qua kịch bản {level}m: Không tìm thấy file {input_path}")
            continue

        print(f"\n--- Đang xử lý kịch bản: {level}m ---")
        gdf = gpd.read_file(input_path)

        # 2. Hàm định nghĩa sức chứa dựa trên diện tích mái nhà (Theo ghi chép của bạn)
        def get_capacity_details(row):
            area = row['area_in_meters']
            risk = row['risk_status']
            
            # Phân loại dựa trên diện tích (Lập luận PostGIS của bạn)
            if area > 200:
                b_type = "Public Shelter" # Điểm công cộng (Trường học, UBND)
                cap = int(area / 2.5)     # 2.5m2/người
            elif area >= 50:
                b_type = "Private House"  # Nhà dân tiêu chuẩn
                cap = int(area / 4.0)     # 4.0m2/người (tính cả tường ngăn)
            else:
                b_type = "Small Structure" # Công trình phụ
                cap = 0
            
            # QUAN TRỌNG: Nếu nhà bị ngập (Nguy cơ Cao/Rất cao) -> Sức chứa = 0
            if "Ngập" in risk:
                cap = 0
            
            return pd.Series([b_type, cap])

        # 3. Áp dụng tính toán
        gdf[['building_type', 'capacity']] = gdf.apply(get_capacity_details, axis=1)

        # 4. Thống kê kết quả cho kịch bản này
        safe_capacity = gdf[gdf['risk_status'] == "An toàn"]['capacity'].sum()
        potential_shelters = gdf[(gdf['building_type'] == "Public Shelter") & (gdf['capacity'] > 0)]
        
        print(f"Xong kịch bản {level}m:")
        print(f"   - Tổng sức chứa an toàn: {safe_capacity:,} người")
        print(f"   - Số điểm sơ tán an toàn (>200m2): {len(potential_shelters)} điểm")

        # 5. Xuất file
        gdf.to_file(output_path, driver='GeoJSON')
        print(f"File: {output_path}")

    print("\nHOÀN THÀNH TẤT CẢ KỊCH BẢN MODULE 1!")

if __name__ == "__main__":
    calculate_capacity_multi()