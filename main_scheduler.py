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
    logging.info("========== BẮT ĐẦU CHU KỲ CẬP NHẬT AI HÀNG NGÀY ==========")
    
    # Gọi API lấy thời tiết và đồng bộ CSDL
    run_script(os.path.join("src", "module2_prediction", "04_flood_data_fusion.py"))
    
    # Kích hoạt AI dự báo ngập lụt & sạt lở (Bước này sẽ trigger SpringBoot)
    run_script(os.path.join("src", "module2_prediction", "06_integrated_risk_prediction.py"))
    
    # Cập nhật lại đồ thị giao thông (MCDM Cost) dựa trên rủi ro mới
    run_script(os.path.join("src", "module3_routing", "08_ahp_weighting.py"))
    
    logging.info("========== KẾT THÚC CHU KỲ CẬP NHẬT ==========")

def setup_schedule():
    # Lên lịch chạy vào một giờ cố định mỗi ngày (ví dụ: 06:00 sáng)
    schedule.every().day.at("06:00").do(daily_ai_pipeline)
    
    # [Tùy chọn] Nếu muốn test ngay, hãy mở comment dòng dưới để chạy mỗi 5 phút:
    schedule.every(5).minutes.do(daily_ai_pipeline)

    logging.info("Hệ thống Scheduler đã khởi động. Đang chờ đến giờ ...")

    while True:
        schedule.run_pending()
        time.sleep(60) 

if __name__ == "__main__":
    # Để chắc chắn hệ thống có dữ liệu khi vừa khởi động, chạy ngay lần đầu tiên
    daily_ai_pipeline() 
    
    # Bắt đầu đi vào vòng lặp chờ
    setup_schedule()