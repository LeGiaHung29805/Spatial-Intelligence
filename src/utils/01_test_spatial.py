import sys
import os

#Python nhìn thấy thư mục gốc 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.spatial_handler import SpatialHandler

# Cấu hình đường dẫn file Shapefile
# Hãy đảm bảo file này có đủ các đuôi .shx, .dbf, .prj đi kèm
SHP_PATH = "data/shapefiles/boundary/BatXat(moi)1.shp" 

try:
    handler = SpatialHandler(SHP_PATH)
    bbox = handler.get_bbox()
    print("--- KẾT QUẢ ---")
    print(f"Toạ độ BBOX Bát Xát: {bbox}")
except Exception as e:
    print(f"Lỗi rồi: {e}")