# Sử dụng Python bản slim để nhẹ, nhưng cần cài thêm thư viện GIS hệ thống
FROM python:3.10-slim

# Cài đặt các thư viện hệ thống cần thiết cho GIS (GDAL, PROJ, GEOS)
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    g++ \
    libproj-dev \
    binutils \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập biến môi trường cho GDAL (rất quan trọng)
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# Copy requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code vào container
COPY . .

# Mặc định container sẽ chạy lệnh này (bạn có thể đổi thành script dự báo Bước 06)
CMD ["python", "src/module2/06_inference.py"]