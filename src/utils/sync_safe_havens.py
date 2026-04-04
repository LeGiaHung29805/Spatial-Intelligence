import logging
import sys
from sqlalchemy import text
from pathlib import Path

# Cấu hình Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Đảm bảo import được module CSDL của bạn
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.CSDL.config.db_config import get_engine

def sync_buildings_to_safe_havens():
    """
    Tự động quét bảng batxat_buildings, tìm các công trình công cộng (Public Shelter, Trường học, UBND)
    và đồng bộ sang bảng batxat_safe_havens làm điểm sơ tán.
    """
    logger.info("BẮT ĐẦU ĐỒNG BỘ DỮ LIỆU ĐIỂM SƠ TÁN THẬT TỪ BẢNG TÒA NHÀ...")
    engine = get_engine()

    sync_query = text("""
        TRUNCATE TABLE batxat_safe_havens RESTART IDENTITY;

        INSERT INTO batxat_safe_havens (name, haven_type, max_capacity, current_occupancy, is_accessible, geom)
        SELECT 
            'Điểm sơ tán Tòa nhà #' || id AS name, -- Tạo tên tạm dựa trên ID tòa nhà
            building_type AS haven_type,
            COALESCE(max_capacity, 100) AS max_capacity, -- Nếu null thì mặc định 100
            0 AS current_occupancy,
            TRUE AS is_accessible,
            ST_Centroid(geom) AS geom -- Hàm lấy tâm của tòa nhà
        FROM batxat_buildings
        WHERE building_type IN ('Public Shelter', 'Trường học', 'UBND', 'Trạm y tế', 'School', 'Hospital')
          AND geom IS NOT NULL;
    """)

    try:
        with engine.begin() as conn:
            conn.execute(sync_query)
            
        # Kiểm tra xem đã đồng bộ được bao nhiêu điểm
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM batxat_safe_havens")).scalar()
            
        logger.info(f" HOÀN TẤT! Đã đồng bộ thành công {count} điểm sơ tán thật vào hệ thống.")
        return True

    except Exception as e:
        logger.error(f" LỖI ĐỒNG BỘ: {e}")
        return False

if __name__ == "__main__":
    sync_buildings_to_safe_havens()