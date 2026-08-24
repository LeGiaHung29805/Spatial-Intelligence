import schedule
import time
import subprocess
import logging
import sys
import os

# Cấu hình log để theo dõi hệ thống chạy ngầm
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SCHEDULER - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scheduler.log",encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def run_script(script_path):
    """Hàm hỗ trợ gọi các script Python bằng subprocess"""
    logging.info(f"Bắt đầu chạy: {script_path}")
    try:
        # Sử dụng sys.executable để đảm bảo dùng đúng Python trong .venv
        result = subprocess.run([sys.executable, script_path], check=True, capture_output=True, text=True)
        logging.info(f"Hoàn thành: {script_path}\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"LỖI KHI CHẠY {script_path}:\n{e.stderr}")

def daily_ai_pipeline():
    """Chuỗi công việc tự động mỗi ngày của AI-Service"""
    logging.info(" BẮT ĐẦU CHU KỲ CẬP NHẬT AI HÀNG NGÀY ")
    
    # Gọi API lấy thời tiết và đồng bộ CSDL
    run_script(os.path.join("src", "module2_prediction", "04_flood_data_fusion.py"))
    
    # Kích hoạt AI dự báo ngập lụt & sạt lở (Bước này sẽ trigger SpringBoot)
    run_script(os.path.join("src", "module2_prediction", "06_integrated_risk_prediction.py"))
    
    # Cập nhật lại đồ thị giao thông (MCDM Cost) dựa trên rủi ro mới
    run_script(os.path.join("src", "module3_routing", "08_ahp_weighting.py"))
    
    logging.info("KẾT THÚC CHU KỲ CẬP NHẬT")


def get_interval_minutes():
    """Trả về chu kỳ pipeline; giữ mặc định 5 phút theo hành vi ban đầu của hệ thống."""
    raw_interval = os.getenv("AI_PIPELINE_INTERVAL_MINUTES", "5").strip()
    if not raw_interval:
        return None
    try:
        interval = int(raw_interval)
        if interval < 1:
            raise ValueError
        return interval
    except ValueError:
        logging.warning(
            "AI_PIPELINE_INTERVAL_MINUTES=%r không hợp lệ; bỏ qua lịch chạy theo phút.",
            raw_interval,
        )
        return None

def setup_schedule():
    # Lên lịch chạy vào một giờ cố định mỗi ngày (ví dụ: 06:00 sáng)
    schedule.every().day.at("06:00").do(daily_ai_pipeline)

    interval = get_interval_minutes()
    if interval:
        schedule.every(interval).minutes.do(daily_ai_pipeline)
        logging.info("Đã bật chu kỳ AI theo phút: mỗi %s phút.", interval)

    logging.info("Hệ thống Scheduler đã khởi động. Đang chờ đến giờ ...")

    while True:
        schedule.run_pending()
        time.sleep(60) 

if __name__ == "__main__":
    # Để chắc chắn hệ thống có dữ liệu khi vừa khởi động, chạy ngay lần đầu tiên
    daily_ai_pipeline() 
    
    # Bắt đầu đi vào vòng lặp chờ
    setup_schedule()
