import rasterio
import numpy as np
import joblib
import os
from rasterio import windows

def generate_landslide_heatmap():
    print("KHỞI ĐỘNG BƯỚC 3: XUẤT BẢN ĐỒ CẢNH BÁO SẠT LỞ (HEATMAP)...")

    dem_path = "data/raw/dem/Copernicus_DEM_BatXat_30m.tif"
    slope_path = "data/raw/slope/Slope_BatXat_DEM_30m.tif"
    model_path = "models/rf_landslide_model.pkl"
    output_heatmap = "data/processed/batxat_landslide_heatmap.tif"

    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy 'Bộ não' tại {model_path}. Hãy chạy file 02 trước!")
        return

    print("Đang nạp mô hình Random Forest vào bộ nhớ...")
    rf_model = joblib.load(model_path)

    print("Đang đồng bộ hóa kích thước hai bản đồ...")
    with rasterio.open(dem_path) as src_dem, rasterio.open(slope_path) as src_slope:
        # Lấy kích thước tối thiểu chung của cả 2 mảng (để gọt những pixel dư thừa)
        min_height = min(src_dem.height, src_slope.height)
        min_width = min(src_dem.width, src_slope.width)
        
        # Đọc 2 bản đồ với khung hình đã gọt
        window = windows.Window(0, 0, min_width, min_height)
        dem_array = src_dem.read(1, window=window)
        slope_array = src_slope.read(1, window=window)
        
        # Sao chép profile của DEM và cập nhật kích thước mới
        profile = src_dem.profile
        profile.update(
            width=min_width,
            height=min_height,
            dtype=rasterio.float32,
            count=1,
            compress='lzw',
            nodata=-9999.0
        )

    # 4. Lọc bỏ các vùng viền đen (NoData)
    valid_mask = (dem_array > 0) & (slope_array >= 0)
    
    dem_valid = dem_array[valid_mask]
    slope_valid = slope_array[valid_mask]
    
    print(f"Tìm thấy {len(dem_valid):,} điểm ảnh hợp lệ để dự báo.")

    if len(dem_valid) == 0:
        print("Lỗi: Không có điểm ảnh hợp lệ nào. Cả hai mảng đều là NoData!")
        return

    # 5. Chuẩn hóa dữ liệu
    print("⚖️ Đang chuẩn hóa dữ liệu bản đồ về thang đo [0, 1]...")
    # Thêm check chống chia cho 0
    dem_range = dem_valid.max() - dem_valid.min()
    slope_range = slope_valid.max() - slope_valid.min()
    
    dem_norm = (dem_valid - dem_valid.min()) / dem_range if dem_range > 0 else np.zeros_like(dem_valid)
    slope_norm = (slope_valid - slope_valid.min()) / slope_range if slope_range > 0 else np.zeros_like(slope_valid)

    X_predict = np.column_stack((dem_norm, slope_norm))

    # 6. AI THỰC HIỆN DỰ BÁO
    print("AI đang chấm điểm xác suất sạt lở cho từng pixel...")
    probabilities = rf_model.predict_proba(X_predict)[:, 1] 

    # 7. Cuộn dữ liệu lại thành Bản đồ (Raster)
    print("Đang vẽ Bản đồ Nhiệt (Heatmap)...")
    heatmap_array = np.full((min_height, min_width), -9999.0, dtype=np.float32)
    heatmap_array[valid_mask] = probabilities

    with rasterio.open(output_heatmap, 'w', **profile) as dst:
        dst.write(heatmap_array, 1)

    print(f"\nHOÀN THÀNH XUẤT SẮC! Bản đồ Cảnh báo đã ra lò tại: {output_heatmap}")
    print(f"Thống kê nhanh mức độ rủi ro:")
    print(f"   - Mức an toàn (< 30%): {np.sum(probabilities < 0.3):,} pixel")
    print(f"   - Mức nguy hiểm (> 70%): {np.sum(probabilities > 0.7):,} pixel")

if __name__ == "__main__":
    generate_landslide_heatmap()