import rasterio
import pandas as pd
import numpy as np
import random
import os
from sklearn.preprocessing import MinMaxScaler

def create_fused_dataset():
    print("KHỞI ĐỘNG MODULE 2: DATA FUSION & TRÍCH XUẤT CÓ ĐIỀU KIỆN...")

    # 1. Khai báo đường dẫn (Đã trỏ thẳng vào file Slope_BatXat_DEM_30m của bạn)
    dem_path = "data/raw/dem/Copernicus_DEM_BatXat_30m.tif"
    slope_path = "data/raw/slope/Slope_BatXat_DEM_30m.tif" 
    output_csv = "data/processed/landslide_training_data.csv"

    if not os.path.exists(slope_path) or not os.path.exists(dem_path):
        print(f"Lỗi: Không tìm thấy file tại thư mục raw! Hãy kiểm tra lại đường dẫn.")
        return

    # 2. ĐỌC MA TRẬN ĐỘ DỐC VÀ TÌM ĐIỂM THEO ĐIỀU KIỆN
    print("Đang quét bản đồ để tìm các khu vực thỏa mãn điều kiện...")
    with rasterio.open(slope_path) as src_slope:
        slope_array = src_slope.read(1)
        transform = src_slope.transform
        
        # Tạo mặt nạ (Mask) cho 2 điều kiện rành mạch
        # Điều kiện 1: An toàn (Độ dốc >= 0 và < 15 độ)
        safe_mask = (slope_array >= 0) & (slope_array < 15)
        
        # Điều kiện 2: Nguy hiểm/Sạt lở (Độ dốc > 30 độ)
        danger_mask = (slope_array > 30)

        # Lấy tọa độ (Row, Col) của các điểm thỏa mãn
        safe_rows, safe_cols = np.where(safe_mask)
        danger_rows, danger_cols = np.where(danger_mask)

        # Lấy ngẫu nhiên đúng 500 điểm cho mỗi loại để Cân bằng dữ liệu (Balanced Dataset)
        # (Dùng min() để phòng trường hợp dữ liệu nhỏ không đủ 500 điểm)
        num_safe = min(500, len(safe_rows))
        num_danger = min(500, len(danger_rows))
        
        safe_indices = random.sample(range(len(safe_rows)), num_safe)
        danger_indices = random.sample(range(len(danger_rows)), num_danger)

        # Chuyển đổi (Row, Col) thành tọa độ không gian (X, Y)
        points = []
        labels = []
        slope_vals = []
        
        for idx in safe_indices:
            r, c = safe_rows[idx], safe_cols[idx]
            x, y = rasterio.transform.xy(transform, r, c)
            points.append((x, y))
            labels.append(0) # Gán nhãn 0 (An toàn)
            slope_vals.append(slope_array[r, c])

        for idx in danger_indices:
            r, c = danger_rows[idx], danger_cols[idx]
            x, y = rasterio.transform.xy(transform, r, c)
            points.append((x, y))
            labels.append(1) # Gán nhãn 1 (Sạt lở)
            slope_vals.append(slope_array[r, c])

    # 3. XUYÊN THỦNG LỚP CAO ĐỘ (DEM) TẠI CÁC ĐIỂM VỪA TÌM ĐƯỢC
    print(f"Đang 'khoan' xuyên lớp Cao trình (Elevation) tại {len(points)} điểm này...")
    with rasterio.open(dem_path) as src_dem:
        dem_vals = [val[0] for val in src_dem.sample(points)]

    # 4. TẠO BẢNG DỮ LIỆU
    df = pd.DataFrame({
        'Coord_X': [p[0] for p in points],
        'Coord_Y': [p[1] for p in points],
        'Elevation': dem_vals,
        'Slope': slope_vals,
        'Landslide_Label': labels
    })

    # Lọc bỏ nhiễu nếu có pixel rỗng ở viền bản đồ (Elevation âm hoặc rất nhỏ)
    df = df[df['Elevation'] > 0].copy()

    # 5. CHUẨN HÓA DỮ LIỆU (Normalization) về thang [0, 1]
    print("⚖️ Đang chuẩn hóa (MinMaxScaler) các biến số về thang đo [0, 1]...")
    scaler = MinMaxScaler()
    df[['Elevation_Norm', 'Slope_Norm']] = scaler.fit_transform(df[['Elevation', 'Slope']])

    # Đảo lại thứ tự cột cho đẹp mắt
    df = df[['Coord_X', 'Coord_Y', 'Elevation', 'Elevation_Norm', 'Slope', 'Slope_Norm', 'Landslide_Label']]

    # 6. XUẤT FILE CHO MÔ HÌNH RANDOM FOREST
    df.to_csv(output_csv, index=False)
    
    print(f"\nHOÀN THÀNH! Tập dữ liệu đã được tối ưu hóa và cân bằng hoàn hảo.")
    print(f"Thống kê Label trong file {output_csv}:")
    print(df['Landslide_Label'].value_counts().rename(index={0: "An toàn (0)", 1: "Sạt lở (1)"}))
    
    print("\nXem thử 3 dòng MẪU AN TOÀN (< 15 độ):")
    print(df[df['Landslide_Label'] == 0][['Slope', 'Slope_Norm', 'Landslide_Label']].head(3))
    print("\nXem thử 3 dòng MẪU NGUY HIỂM (> 30 độ):")
    print(df[df['Landslide_Label'] == 1][['Slope', 'Slope_Norm', 'Landslide_Label']].head(3))

if __name__ == "__main__":
    create_fused_dataset()