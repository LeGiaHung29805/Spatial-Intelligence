import geopandas as gpd
import pandas as pd
import rasterio
import requests
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

def flood_data_fusion_2years():
    print("BƯỚC 4: THỰC HIỆN DATA FUSION (2 NĂM DỮ LIỆU)...")

    # --- CẤU HÌNH ---
    API_KEY = "97JQA9YSSKCDXW3HVV8CZDQB6" 
    SHP_PATH = "data/shapefiles/boundary/BatXat(moi)1.shp"
    DEM_PATH = "data/raw/dem/Copernicus_DEM_BatXat_30m.tif"
    SLOPE_PATH = "data/raw/slope/Slope_BatXat_DEM_30m.tif"
    LC_PATH = "data/raw/landcover/LandCover_BatXat_DynamicWorld_10m.tif"
    OUTPUT_CSV = "data/processed/flood_training_data_lstm.csv"

    # 1. Tự động tính toán mốc 2 năm (730 ngày)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    print(f"Chu kỳ lấy dữ liệu: Từ {start_date} đến {end_date}")

    # 2. Lấy tọa độ tâm huyện Bát Xát
    gdf = gpd.read_file(SHP_PATH).to_crs(epsg=4326)
    centroid = gdf.geometry.centroid.iloc[0]
    lat, lon = centroid.y, centroid.x

    # 3. Gọi API lấy mưa (Visual Crossing cho phép lấy 1000 records/ngày nên 730 ngày là ổn)
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{start_date}/{end_date}"
    params = {'unitGroup': 'metric', 'key': API_KEY, 'include': 'days', 'elements': 'datetime,precip'}
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Lỗi API ({response.status_code}). Có thể do hết quota hoặc sai Key.")
        return
    
    df_rain = pd.DataFrame(response.json()['days'])

    # 4. Trích xuất đặc trưng tĩnh từ các file TIF của bạn
    print("Trộn dữ liệu tĩnh (DEM, Slope, LandCover)...")
    with rasterio.open(DEM_PATH) as dem, rasterio.open(SLOPE_PATH) as slope, rasterio.open(LC_PATH) as lc:
        row, col = dem.index(lon, lat)
        val_dem = dem.read(1)[row, col]
        val_slope = slope.read(1)[row, col]
        val_lc = lc.read(1)[row, col]

    # 5. Gán dữ liệu tĩnh vào toàn bộ 730 dòng dữ liệu động
    df_rain['elevation'] = val_dem
    df_rain['slope'] = val_slope
    df_rain['landcover'] = val_lc # Mã số từ Module 1 của bạn
    
    # Tạo biến Mực nước giả định (Water Proxy)
    # Với 2 năm, biến này sẽ thể hiện rõ rệt sự khác biệt giữa các mùa
    df_rain['water_proxy'] = df_rain['precip'].rolling(window=3).sum().fillna(0)

    # 6. Chuẩn hóa [0, 1] cho toàn bộ tập dữ liệu lớn
    # scaler = MinMaxScaler()
    # cols_to_scale = ['precip', 'water_proxy', 'elevation', 'slope']
    # df_rain[cols_to_scale] = scaler.fit_transform(df_rain[cols_to_scale])

    df_rain.to_csv(OUTPUT_CSV, index=False)
    print(f"HOÀN THÀNH! Đã xử lý {len(df_rain)} ngày dữ liệu tại: {OUTPUT_CSV}")

if __name__ == "__main__":
    flood_data_fusion_2years()