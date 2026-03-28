import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def audit_flood_data():
    DATA_DIR = "data/processed"
    file_path = os.path.join(DATA_DIR, "flood_full_history.csv")
    
    if not os.path.exists(file_path):
        print("Không tìm thấy file dữ liệu!")
        return
        
    df = pd.read_csv(file_path)
    # Xác định nhãn lũ (Giống logic trong code train của bạn)
    df['is_flood'] = ((df['water_level'] > df['water_level'].quantile(0.90)) | (df['is_event'] == 1)).astype(int)

    print("THỐNG KÊ LƯỢNG MƯA TẠI BÁT XÁT:")
    stats = df.groupby('is_flood')['precip_BAT_XAT'].agg(['mean', 'max', 'std', 'count']).rename(
        index={0: 'Ngày thường', 1: 'Ngày lũ'}
    )
    print(stats)

    #KIỂM TRA ĐỘ TƯƠNG QUAN (CORRELATION)
    correlation = df['precip_BAT_XAT'].corr(df['is_flood'])
    print(f"\nHệ số tương quan giữa Mưa và Lũ: {correlation:.4f}")

    #TRỰC QUAN HÓA ĐỘ TƯƠNG PHẢN
    plt.figure(figsize=(12, 5))

    # Biểu đồ Boxplot (Để xem sự phân tách)
    plt.subplot(1, 2, 1)
    sns.boxplot(x='is_flood', y='precip_BAT_XAT', data=df, palette='Set2')
    plt.title('So sánh lượng mưa: Ngày thường vs Ngày lũ')
    plt.xticks([0, 1], ['Ngày thường', 'Ngày lũ'])
    plt.ylabel('Lượng mưa (mm)')

    # Biểu đồ Phân phối (Density Plot)
    plt.subplot(1, 2, 2)
    sns.kdeplot(df[df['is_flood']==0]['precip_BAT_XAT'], label='Ngày thường', fill=True)
    sns.kdeplot(df[df['is_flood']==1]['precip_BAT_XAT'], label='Ngày lũ', fill=True)
    plt.title('Phân phối mật độ lượng mưa')
    plt.xlabel('Lượng mưa (mm)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 4. KIỂM TRA TÍN HIỆU CẢNH BÁO SỚM (LAG ANALYSIS)
    print("\nKIỂM TRA TÍN HIỆU TRƯỚC LŨ 1-3 NGÀY:")
    for lag in range(1, 4):
        lag_corr = df['precip_BAT_XAT'].shift(lag).corr(df['is_flood'])
        print(f"   - Tương quan giữa mưa trước {lag} ngày và lũ hôm nay: {lag_corr:.4f}")

if __name__ == "__main__":
    audit_flood_data()