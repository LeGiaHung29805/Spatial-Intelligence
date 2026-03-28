import rasterio
import numpy as np
import matplotlib.pyplot as plt

def calculate_batxat_slope(dem_path):
    print(f"🏔️ Đang phân tích địa hình từ file: {dem_path}")
    
    with rasterio.open(dem_path) as src:
        # 1. Đọc dữ liệu cao độ
        dem = src.read(1).astype(float)
        # Thay thế các giá trị NoData bằng NaN để không làm sai lệch kết quả
        dem[dem < -9000] = np.nan 
        
        # Lấy độ phân giải pixel (thường là 30m cho Copernicus DEM)
        res = src.res[0] 
        
        # 2. Thuật toán tính Gradient (độ chênh lệch) theo 2 chiều X và Y
        # Chênh lệch độ cao / khoảng cách 
        dx, dy = np.gradient(dem, res)
        
        # 3. Tính độ dốc (Slope) theo đơn vị Degree (Độ)
        slope = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_deg = np.rad2deg(slope)
        
        # 4. Tính toán các chỉ số thống kê
        mean_slope = np.nanmean(slope_deg)
        max_slope = np.nanmax(slope_deg)
        
        print("-" * 30)
        print(f"📊 KẾT QUẢ PHÂN TÍCH ĐỊA HÌNH:")
        print(f"📍 Độ dốc trung bình: {mean_slope:.2f} độ")
        print(f"📍 Độ dốc lớn nhất:   {max_slope:.2f} độ")
        print("-" * 30)
        
        # 5. HIỂN THỊ BẢN ĐỒ ĐỂ BẠN THẤY (Dùng cho báo cáo)
        plt.figure(figsize=(10, 6))
        plt.imshow(slope_deg, cmap='terrain')
        plt.colorbar(label='Độ dốc (Degrees)')
        plt.title(f"Bản đồ độ dốc huyện Bát Xát\n(Mean: {mean_slope:.2f}°)")
        plt.show()
        
        return mean_slope

if __name__ == "__main__":
    DEM_FILE = "data/raw/dem/Copernicus_DEM_BatXat_30m.tif"
    mean_val = calculate_batxat_slope(DEM_FILE)