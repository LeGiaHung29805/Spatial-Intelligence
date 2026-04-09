import numpy as np

# 1. Kịch bản SAFETY (Ưu tiên sinh mạng - Né 100% rủi ro)
matrix_safety = np.array([
    [1,     1/7,    1/9,    1/5,    1/6,    1/5],  # Khoảng cách ít quan trọng nhất
    [7,     1,      1/2,    3,      2,      2  ],  # Ngập lụt (Rất quan trọng)
    [9,     2,      1,      5,      3,      3  ],  # Sạt lở (Quan trọng NHẤT)
    [5,     1/3,    1/5,    1,      1/2,    1/2],  # Hạng đường
    [6,     1/2,    1/3,    2,      1,      1  ],  # Cầu yếu
    [5,     1/2,    1/3,    2,      1,      1  ]   # Cảnh báo cộng đồng
])

# 2. Kịch bản RESCUE (Cứu hộ khẩn cấp - Chấp nhận rủi ro nhẹ để cứu người nhanh)
matrix_rescue = np.array([
    [1,     2,      1/3,    1/4,    1/2,    1  ],  # Khoảng cách được nâng tầm quan trọng lên
    [1/2,   1,      1/4,    1/5,    1/3,    1/2],  # Ngập lụt (Chấp nhận lội nước nông)
    [3,     4,      1,      1,      2,      2  ],  # Sạt lở (Vẫn phải né)
    [4,     5,      1,      1,      2,      3  ],  # Hạng đường (Quan trọng vì đi xe chuyên dụng)
    [2,     3,      1/2,    1/2,    1,      1  ],  # Cầu yếu
    [1,     2,      1/2,    1/3,    1,      1  ]   # Cảnh báo cộng đồng
])

def test_ahp_consistency(matrix, strategy_name):
    print(f"\n[{strategy_name.upper()}] - KIỂM ĐỊNH TÍNH NHẤT QUÁN TOÁN HỌC MA TRẬN AHP")
    print("-" * 65)
    n = matrix.shape[0]
    
    # 1. Tính trọng số (Weights)
    column_sums = matrix.sum(axis=0)
    normalized_matrix = matrix / column_sums
    weights = normalized_matrix.mean(axis=1)
    
    # 2. Tính Tỷ số nhất quán (CR)
    weighted_sum = np.dot(matrix, weights)
    lambda_max = (weighted_sum / weights).mean()
    
    ci = (lambda_max - n) / (n - 1)
    
    # Chỉ số ngẫu nhiên RI của Saaty cho ma trận 6x6
    ri = 1.24 
    cr = ci / ri
    
    # In kết quả định dạng đẹp
    criteria = ['Khoảng cách', 'Ngập lụt', 'Sạt lở', 'Hạng đường', 'Cầu yếu', 'Cộng đồng']
    for i in range(len(weights)):
        print(f" - Trọng số [{criteria[i]}]: \t{weights[i]*100:.2f}%")
        
    print(f"\n => Trị riêng cực đại (Lambda Max) : {lambda_max:.4f}")
    print(f" => Tỷ số nhất quán (CR)           : {cr:.4f}")
    
    if cr < 0.1:
        print(" => KẾT LUẬN                       : [HỢP LỆ] Ma trận nhất quán tốt (CR < 0.1).")
    else:
        print(" => KẾT LUẬN                       : [LỖI] Ma trận mâu thuẫn (CR >= 0.1). Cần tinh chỉnh!")

if __name__ == "__main__":
    test_ahp_consistency(matrix_safety, "Chiến lược Safety (Sơ tán người dân)")
    test_ahp_consistency(matrix_rescue, "Chiến lược Rescue (Điều phối cứu hộ)")