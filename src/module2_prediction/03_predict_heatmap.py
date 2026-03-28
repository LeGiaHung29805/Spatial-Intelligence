import rasterio
import numpy as np
import joblib
import os
import geopandas as gpd
from rasterio import features
from scipy.ndimage import distance_transform_edt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def generate_landslide_heatmap_enhanced():
    print("KHỞI ĐỘNG BƯỚC 3: XUẤT BẢN ĐỒ NHIỆT (INFERENCE - 5 FEATURES)...")

    # Đường dẫn dữ liệu
    dem_path = "data/raw/dem/Copernicus_DEM_BatXat_30m.tif"
    slope_path = "data/raw/slope/Slope_BatXat_DEM_30m.tif"
    river_path = "data/shapefiles/hydrology/Song_ngoi_Bat_Xat.shp"
    lake_path = "data/shapefiles/hydrology/Ho_Bat_Xat.shp"
    
    model_path = "models/rf_landslide_model.pkl"
    scaler_path = "models/scaler.pkl" 
    output_heatmap = "data/processed/batxat_landslide_heatmap.tif"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Lỗi: Thiếu 'Bộ não' AI hoặc 'Thước đo' (Scaler).")
        return

    # Load AI Model và Scaler (Đã train với 5 features)
    rf_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with rasterio.open(dem_path) as src_dem, rasterio.open(slope_path) as src_slope:
        dem_array = src_dem.read(1)
        slope_array = src_slope.read(1)
        
        min_height = min(dem_array.shape[0], slope_array.shape[0])
        min_width = min(dem_array.shape[1], slope_array.shape[1])
        
        dem_array = dem_array[:min_height, :min_width]
        slope_array = slope_array[:min_height, :min_width]
        
        transform = src_dem.transform
        profile = src_dem.profile
        pixel_size = transform[0] 

    profile.update(height=min_height, width=min_width, dtype=rasterio.float32, nodata=-9999.0)

    #TẠO LỚP KHOẢNG CÁCH THỦY VĂN (EDT)
    print("💧 Đang số hóa mạng lưới sông ngòi...")
    rivers = gpd.read_file(river_path).to_crs(profile['crs'])
    lakes = gpd.read_file(lake_path).to_crs(profile['crs'])
    water_geoms = [(geom, 1) for geom in rivers.geometry] + [(geom, 1) for geom in lakes.geometry]
    
    water_mask = features.rasterize(
        water_geoms, out_shape=(min_height, min_width),
        transform=transform, fill=0, all_touched=True
    )
    dist_array = distance_transform_edt(water_mask == 0) * pixel_size

    # GIẢ LẬP KỊCH BẢN MƯA VÀ ĐẤT
    print("Đang thiết lập kịch bản: Mưa lớn 200mm & Độ ẩm đất cao...")
    # Tạo mảng có cùng kích thước với bản đồ, lấp đầy bằng giá trị kịch bản
    rain_scenario = np.full((min_height, min_width), 200.0) # Giả lập mưa 200mm
    soil_scenario = np.full((min_height, min_width), 0.7)   # Giả lập độ ẩm đất 0.7

    # TRÍCH XUẤT ĐẶC TRƯNG & CHUẨN HÓA (Phải đúng thứ tự Step 2)
    # Thứ tự features: ['Slope', 'Elevation', 'Dist_to_Water', 'precip_3d_BAT_XAT', 'soil_BAT_XAT']
    print("Đang áp dụng Thước đo Scaler cho 5 lớp dữ liệu...")
    valid_mask = (dem_array > -100) & (slope_array >= 0) 
    
    X_raw = np.column_stack((
        slope_array[valid_mask], 
        dem_array[valid_mask], 
        dist_array[valid_mask],
        rain_scenario[valid_mask],
        soil_scenario[valid_mask]
    ))
    
    X_scaled = scaler.transform(X_raw)

    print('AI đang quét và "vẽ" Bản đồ nhiệt (Dynamic Landslide Heatmap)...')
    probabilities = rf_model.predict_proba(X_scaled)[:, 1] 

    heatmap_array = np.full((min_height, min_width), -9999.0, dtype=np.float32)
    heatmap_array[valid_mask] = probabilities 

    with rasterio.open(output_heatmap, 'w', **profile) as dst:
        dst.write(heatmap_array, 1)

    print(f"THÀNH CÔNG! Bản đồ nhiệt 5 lớp đã lưu tại: {output_heatmap}")

if __name__ == "__main__":
    generate_landslide_heatmap_enhanced()