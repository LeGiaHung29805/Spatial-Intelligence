import rasterio
import pandas as pd
import numpy as np
import random
import os
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler

def create_fused_dataset_with_hydro():
    print("KHỞI ĐỘNG MODULE 2: DATA FUSION (TÍCH HỢP THỦY VĂN)...")

    dem_path = "data/raw/dem/Copernicus_DEM_BatXat_30m_New.tif"
    slope_path = "data/raw/slope/Slope_BatXat_DEM_30m_New.tif" 
    river_path = "data/shapefiles/hydrology/Song_ngoi_Bat_Xat_1.shp"
    lake_path = "data/shapefiles/hydrology/Ho_Bat_Xat_1.shp"
    output_csv = "data/processed/landslide_training_data.csv"

    #ĐỌC DỮ LIỆU ĐỘ DỐC & LẤY ĐIỂM MẪU 
    with rasterio.open(slope_path) as src_slope:
        slope_array = src_slope.read(1)
        transform = src_slope.transform
        safe_mask = (slope_array >= 0) & (slope_array < 15)
        danger_mask = (slope_array > 30)
        safe_rows, safe_cols = np.where(safe_mask)  
        danger_rows, danger_cols = np.where(danger_mask)

        num_safe = min(500, len(safe_rows))
        num_danger = min(500, len(danger_rows))
        
        safe_indices = random.sample(range(len(safe_rows)), num_safe)
        danger_indices = random.sample(range(len(danger_rows)), num_danger)

        points_coords = []
        labels = []
        slope_vals = []
        
        for idx in safe_indices:
            r, c = safe_rows[idx], safe_cols[idx]
            x, y = rasterio.transform.xy(transform, r, c)
            points_coords.append((x, y))
            labels.append(0)
            slope_vals.append(slope_array[r, c])

        for idx in danger_indices:
            r, c = danger_rows[idx], danger_cols[idx]
            x, y = rasterio.transform.xy(transform, r, c)
            points_coords.append((x, y))
            labels.append(1)
            slope_vals.append(slope_array[r, c])

    #TRÍCH XUẤT CAO ĐỘ (DEM)
    with rasterio.open(dem_path) as src_dem:
        dem_vals = [val[0] for val in src_dem.sample(points_coords)]

    #TÍCH HỢP THỦY VĂN (TÍNH KHOẢNG CÁCH ĐẾN NƯỚC)
    print("Đang tính toán yếu tố Thủy văn cho các điểm mẫu...")
    rivers = gpd.read_file(river_path)
    lakes = gpd.read_file(lake_path)
    
    # Gom tất cả sông hồ thành 1 khối để tính khoảng cách
    all_water = pd.concat([rivers.geometry, lakes.geometry]).unary_union
    
    # Tạo GeoDataFrame cho các điểm mẫu để tính toán không gian
    gdf_points = gpd.GeoDataFrame(
        {'geometry': [Point(p[0], p[1]) for p in points_coords]}, 
        crs=rivers.crs
    )
    
    # Chuyển sang hệ mét (VD: EPSG:32648) để tính khoảng cách mét chính xác
    gdf_points_m = gdf_points.to_crs(epsg=32648)
    water_m = gpd.GeoSeries([all_water], crs=rivers.crs).to_crs(epsg=32648).iloc[0]
    
    dist_to_water = gdf_points_m.distance(water_m)

    #TẠO BẢNG DỮ LIỆU TỔNG HỢP 
    df = pd.DataFrame({
        'Coord_X': [p[0] for p in points_coords],
        'Coord_Y': [p[1] for p in points_coords],
        'Elevation': dem_vals,
        'Slope': slope_vals,
        'Dist_to_Water': dist_to_water,
        'Landslide_Label': labels
    })

    #CHUẨN HÓA (Normalization)
    # scaler = MinMaxScaler()
    # df[['Elevation_Norm', 'Slope_Norm', 'Dist_Water_Norm']] = scaler.fit_transform(
    #     df[['Elevation', 'Slope', 'Dist_to_Water']]
    # )

    df.to_csv(output_csv, index=False)
    print(f"Hoàn thành Fused Dataset với {len(df)} mẫu và 3 đặc trưng (Slope, Elevation, Dist_Water).")

if __name__ == "__main__":
    create_fused_dataset_with_hydro()