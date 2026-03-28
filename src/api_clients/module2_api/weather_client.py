import requests
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Đưa Key ra đây (Thực tế sau này nên đọc từ file .env)
API_KEY_VC = "97JQA9YSSKCDXW3HVV8CZDQB6"

def fetch_vc_history_range(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """Lấy dữ liệu lượng mưa từ Visual Crossing Timeline API"""
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{start_date}/{end_date}"
    params = {
        'unitGroup': 'metric',
        'key': API_KEY_VC,
        'include': 'days',
        'elements': 'datetime,precip'
    }
    try:
        res = requests.get(url, params=params, timeout=30)
        if res.status_code == 200:
            data = res.json()
            return {day['datetime']: day['precip'] for day in data['days']}
        else:
            logger.warning(f"API VC báo lỗi {res.status_code} cho {start_date} -> {end_date}")
            return {}
    except Exception as e:
        logger.error(f"Lỗi kết nối API Visual Crossing: {e}")
        return {}

def fetch_soil_fatigue_api(lat: float, lon: float, past_days: int = 15) -> dict:
    """Gọi API vệ tinh Open-Meteo lấy Độ ẩm đất tầng sâu -> Độ mỏi đất"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": past_days,
        "forecast_days": 1, 
        "hourly": "soil_moisture_27_to_81cm", 
        "timezone": "Asia/Bangkok"
    }
    try:
        res = requests.get(url, params=params, timeout=30)
        if res.status_code == 200:
            data = res.json()
            times = pd.to_datetime(data['hourly']['time'])
            fatigue_vals = data['hourly']['soil_moisture_27_to_81cm']
            
            temp_df = pd.DataFrame({'datetime': times, 'fatigue': fatigue_vals})
            temp_df['date'] = temp_df['datetime'].dt.strftime('%Y-%m-%d')
            return temp_df.groupby('date')['fatigue'].mean().to_dict()
        else:
            logger.warning(f"API Soil Fatigue báo lỗi {res.status_code}")
            return {}
    except Exception as e:
        logger.error(f"Lỗi kết nối API Open-Meteo: {e}")
        return {}