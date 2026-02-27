import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.spatial_handler import SpatialHandler

SHP_PATH = "data/shapefiles/boundary/BatXat(moi)1.shp" 

try:
    handler = SpatialHandler(SHP_PATH)
    bbox = handler.get_bbox()
    print("--- KẾT QUẢ ---")
    print(f"Toạ độ BBOX Bát Xát: {bbox}")
except Exception as e:
    print(f"Lỗi rồi: {e}")