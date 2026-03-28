import requests
import logging

# Cấu hình logging
logger = logging.getLogger(__name__)

SPRING_BOOT_URL = "http://localhost:8080/api/v1"

def trigger_heatmap_update() -> bool:
    """
    Client gọi API nội bộ của Spring Boot để yêu cầu phát sóng WebSocket bản đồ nhiệt.
    """
    endpoint = f"{SPRING_BOOT_URL}/map/internal/trigger-broadcast"
    logger.info(f"[Spring Boot Client] Đang gửi lệnh tới: {endpoint}")
    
    try:
        # Timeout 5s để Python không bị "treo" nếu Spring Boot đang tắt
        res = requests.post(endpoint, timeout=5)
        
        if res.status_code == 200:
            logger.info("[Spring Boot Client] Thành công! Bản đồ Web đang được cập nhật Real-time.")
            return True
        else:
            logger.warning(f"[Spring Boot Client] Cảnh báo từ Server: Mã lỗi {res.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.error("[Spring Boot Client] Lỗi kết nối: Spring Boot đang tắt hoặc sai Port.")
        return False
    except Exception as e:
        logger.error(f"[Spring Boot Client] Lỗi không xác định: {e}")
        return False