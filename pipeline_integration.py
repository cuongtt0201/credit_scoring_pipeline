"""
pipeline_integration.py
Module tích hợp dữ liệu đa nguồn và kỹ nghệ đặc trưng (Feature Engineering)
Tuân thủ nguyên tắc Point-in-time (chặn rò rỉ dữ liệu).
"""

import pandas as pd
import numpy as np

def build_features(train_path='data/applications_train.csv',
                   test_path='data/applications_test.csv',
                   customers_path='data/customers.csv',
                   dealers_path='data/dealers.csv',
                   bureau_path='data/credit_bureau.csv',
                   repayment_path='data/repayment_history.csv',
                   payroll_path='data/payroll.csv'):
    
    print("[1/5] Đang tải dữ liệu thô...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_cust = pd.read_csv(customers_path)
    df_dealers = pd.read_csv(dealers_path)
    df_bureau = pd.read_csv(bureau_path)
    df_repay = pd.read_csv(repayment_path)
    df_payroll = pd.read_csv(payroll_path)

    # Đánh dấu nguồn train / test để xử lý chung rồi tách lại
    df_train['is_test'] = 0
    df_test['is_test'] = 1
    df_test['vo_no'] = np.nan
    
    apps = pd.concat([df_train, df_test], ignore_index=True)
    n_total = len(apps)
    print(f"-> Tổng số hồ sơ: {n_total:,} (Train: {len(df_train):,}, Test: {len(df_test):,})")

    # -------------------------------------------------------------
    # 2. XỬ LÝ & GOM CỤM CREDIT BUREAU (1-N theo ma_ho_so)
    # -------------------------------------------------------------
    print("[2/5] Đang xử lý và gom cụm Credit Bureau...")
    bureau_agg = df_bureau.groupby('ma_ho_so').agg(
        bureau_so_khoan_vay=('du_no_hien_tai', 'count'),
        bureau_tong_du_no=('du_no_hien_tai', 'sum'),
        bureau_tong_tra_hang_thang=('tra_hang_thang', 'sum'),
        bureau_nhom_no_max=('nhom_no_cao_nhat', 'max'),
        bureau_tong_qua_han_12t=('so_lan_qua_han_12t', 'sum')
    ).reset_index()

    apps = apps.merge(bureau_agg, on='ma_ho_so', how='left')
    apps['has_bureau'] = apps['bureau_so_khoan_vay'].notnull().astype(int)
    apps['bureau_so_khoan_vay'] = apps['bureau_so_khoan_vay'].fillna(0)
    apps['bureau_tong_du_no'] = apps['bureau_tong_du_no'].fillna(0)
    apps['bureau_tong_tra_hang_thang'] = apps['bureau_tong_tra_hang_thang'].fillna(0)
    apps['bureau_nhom_no_max'] = apps['bureau_nhom_no_max'].fillna(0)
    apps['bureau_tong_qua_han_12t'] = apps['bureau_tong_qua_han_12t'].fillna(0)

    # -------------------------------------------------------------
    # 3. XỬ LÝ & GOM CỤM REPAYMENT HISTORY (1-N theo ma_khach_hang)
    # -------------------------------------------------------------
    print("[3/5] Đang xử lý và gom cụm Lịch sử trả nợ nội bộ TNF...")
    df_repay['ngay_den_han'] = pd.to_datetime(df_repay['ngay_den_han'])
    df_repay['ngay_thanh_toan'] = pd.to_datetime(df_repay['ngay_thanh_toan'])
    
    # Tính số ngày trễ (DPD) và cờ trả thiếu tiền
    df_repay['dpd'] = (df_repay['ngay_thanh_toan'] - df_repay['ngay_den_han']).dt.days
    df_repay['dpd_duong'] = df_repay['dpd'].clip(lower=0)
    df_repay['is_late'] = (df_repay['dpd'] > 0).astype(int)
    df_repay['is_underpaid'] = (df_repay['so_tien_da_tra'] < df_repay['so_tien_phai_tra']).astype(int)

    repay_agg = df_repay.groupby('ma_khach_hang').agg(
        repay_so_ky_vay=('ky_thu', 'count'),
        repay_dpd_max=('dpd_duong', 'max'),
        repay_dpd_mean=('dpd_duong', 'mean'),
        repay_so_lan_tre=('is_late', 'sum'),
        repay_ty_le_tre=('is_late', 'mean'),
        repay_so_lan_thieu=('is_underpaid', 'sum'),
        repay_ty_le_thieu=('is_underpaid', 'mean')
    ).reset_index()

    apps = apps.merge(repay_agg, on='ma_khach_hang', how='left')
    apps['is_existing_cust'] = apps['repay_so_ky_vay'].notnull().astype(int)
    apps['repay_so_ky_vay'] = apps['repay_so_ky_vay'].fillna(0)
    apps['repay_dpd_max'] = apps['repay_dpd_max'].fillna(0)
    apps['repay_dpd_mean'] = apps['repay_dpd_mean'].fillna(0)
    apps['repay_so_lan_tre'] = apps['repay_so_lan_tre'].fillna(0)
    apps['repay_ty_le_tre'] = apps['repay_ty_le_tre'].fillna(0)
    apps['repay_so_lan_thieu'] = apps['repay_so_lan_thieu'].fillna(0)
    apps['repay_ty_le_thieu'] = apps['repay_ty_le_thieu'].fillna(0)

    # -------------------------------------------------------------
    # 4. XỬ LÝ PAYROLL CHẶN THỜI GIAN (Point-in-time theo ngay_nop)
    # -------------------------------------------------------------
    print("[4/5] Đang xử lý Payroll theo nguyên tắc Point-in-time...")
    apps['thang_nop'] = apps['ngay_nop'].str.slice(0, 7) # YYYY-MM
    
    # Merge apps với payroll theo ma_khach_hang rồi lọc thang <= thang_nop
    app_cust_dates = apps[['ma_ho_so', 'ma_khach_hang', 'thang_nop']].copy()
    payroll_merged = df_payroll.merge(app_cust_dates, on='ma_khach_hang', how='inner')
    
    # CHẶN THỜI GIAN: Chỉ giữ các tháng trước hoặc bằng tháng nộp hồ sơ
    payroll_valid = payroll_merged[payroll_merged['thang'] <= payroll_merged['thang_nop']].copy()
    
    # Sắp xếp để lấy tháng gần nhất
    payroll_valid = payroll_valid.sort_values(['ma_ho_so', 'thang'])
    
    # Gom cụm theo ma_ho_so
    payroll_agg = payroll_valid.groupby('ma_ho_so').agg(
        payroll_so_thang=('luong_thuc_nhan', 'count'),
        payroll_luong_gan_nhat=('luong_thuc_nhan', 'last'),
        payroll_luong_mean=('luong_thuc_nhan', 'mean'),
        payroll_luong_std=('luong_thuc_nhan', 'std')
    ).reset_index()

    apps = apps.merge(payroll_agg, on='ma_ho_so', how='left')
    apps['has_payroll'] = apps['payroll_so_thang'].notnull().astype(int)
    apps['payroll_luong_std'] = apps['payroll_luong_std'].fillna(0)

    # -------------------------------------------------------------
    # 5. MERGE CRM & DEALERS + TẠO BIẾN LIÊN NGUỒN (FEATURE ENGINEERING)
    # -------------------------------------------------------------
    print("[5/5] Hoàn thiện tích hợp CRM, Đại lý và biến liên nguồn...")
    # Merge CRM
    apps = apps.merge(df_cust, on='ma_khach_hang', how='left', suffixes=('', '_cust'))
    # Merge Dealers
    df_dealers_renamed = df_dealers.rename(columns={'tinh': 'tinh_dai_ly', 'loai_dai_ly': 'loai_hang_dai_ly'})
    apps = apps.merge(df_dealers_renamed, on='ma_dai_ly', how='left')

    # Tính tuổi
    apps['ngay_nop_dt'] = pd.to_datetime(apps['ngay_nop'])
    apps['ngay_sinh_dt'] = pd.to_datetime(apps['ngay_sinh'])
    apps['tuoi'] = ((apps['ngay_nop_dt'] - apps['ngay_sinh_dt']).dt.days / 365.25).round(1)

    # Biến ứng dụng & khoản vay
    apps['ty_le_tra_truoc'] = apps['tra_truoc'] / apps['gia_tri_hang']
    apps['ap_luc_tra_gop_tnf'] = apps['tra_gop_thang'] / apps['thu_nhap_khai_bao']

    # Thu nhập ước tính (Dùng lương thực nhận gần nhất nếu có payroll, ngược lại dùng thu nhập khai báo)
    apps['thu_nhap_uoc_tinh'] = np.where(apps['has_payroll'] == 1,
                                         apps['payroll_luong_gan_nhat'],
                                         apps['thu_nhap_khai_bao'])

    # BIẾN LIÊN NGUỒN 1: DTI TỔNG (Total Debt-to-Income)
    # (Trả góp mới TNF + Nghĩa vụ trả hàng tháng ở Bureau) / Thu nhập ước tính
    apps['dti_tong'] = (apps['tra_gop_thang'] + apps['bureau_tong_tra_hang_thang']) / apps['thu_nhap_uoc_tinh']

    # BIẾN LIÊN NGUỒN 2: ĐỘ PHÓNG ĐẠI THU NHẬP (Income Exaggeration)
    # 1 - (Thu nhập thực nhận / Thu nhập khai báo). Nếu không có payroll thì = 0 (kèm cờ has_payroll=0)
    apps['do_phong_dai_thu_nhap'] = np.where(
        apps['has_payroll'] == 1,
        1.0 - (apps['payroll_luong_gan_nhat'] / apps['thu_nhap_khai_bao']),
        0.0
    )

    # Đảm bảo số dòng không bị biến đổi
    assert len(apps) == n_total, f"Lỗi số dòng: Ban đầu {n_total}, sau merge {len(apps)}"
    
    # Tách lại Train và Test
    train_processed = apps[apps['is_test'] == 0].drop(columns=['is_test']).copy()
    test_processed = apps[apps['is_test'] == 1].drop(columns=['is_test', 'vo_no']).copy()

    print(f"-> Hoàn tất tích hợp thành công:")
    print(f"   Train kích thước: {train_processed.shape}")
    print(f"   Test kích thước:  {test_processed.shape}")

    return train_processed, test_processed

if __name__ == '__main__':
    train_df, test_df = build_features()
    train_df.to_parquet('data/train_processed.parquet', index=False)
    test_df.to_parquet('data/test_processed.parquet', index=False)
    print("Đã lưu kết quả thành công vào data/train_processed.parquet và test_processed.parquet!")
