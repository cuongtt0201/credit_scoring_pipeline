"""
train_evaluate.py
Module huấn luyện mô hình Logistic Regression với Stratified 5-Fold Cross-Validation,
thực hiện Ablation Study (5 tầng nguồn dữ liệu) và xuất file dự đoán du_doan.csv.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import os

def calculate_ks(y_true, y_prob):
    """Tính chỉ số Kolmogorov-Smirnov (KS statistic)"""
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df = df.sort_values(by='y_prob', ascending=False).reset_index(drop=True)
    df['bad'] = df['y_true']
    df['good'] = 1 - df['y_true']
    df['cum_bad'] = df['bad'].cumsum() / df['bad'].sum()
    df['cum_good'] = df['good'].cumsum() / df['good'].sum()
    ks_stat = (df['cum_bad'] - df['cum_good']).abs().max()
    return ks_stat

def get_feature_sets():
    """Định nghĩa 5 tầng đặc trưng cho Ablation Study theo đúng yêu cầu đề bài"""
    
    # Tầng 1: Chỉ hồ sơ vay tự khai
    stage1_num = ['gia_tri_hang', 'tra_truoc', 'so_tien_vay', 'ky_han_thang', 
                  'lai_suat_nam', 'tra_gop_thang', 'thu_nhap_khai_bao']
    stage1_cat = ['loai_san_pham', 'nghe_nghiep']
    
    # Tầng 2: + Khách hàng (CRM) & Đại lý
    stage2_num = stage1_num + ['so_nguoi_phu_thuoc', 'so_nam_cu_tru', 'so_nam_hop_tac', 'tuoi', 'ty_le_tra_truoc', 'ap_luc_tra_gop_tnf']
    stage2_cat = stage1_cat + ['tinh', 'tinh_trang_hon_nhan', 'loai_nha_o', 'quy_mo', 'loai_hang_dai_ly']
    
    # Tầng 3: + Trung tâm thông tin tín dụng (Credit Bureau)
    stage3_num = stage2_num + ['has_bureau', 'bureau_so_khoan_vay', 'bureau_tong_du_no', 
                               'bureau_tong_tra_hang_thang', 'bureau_nhom_no_max', 'bureau_tong_qua_han_12t']
    stage3_cat = stage2_cat
    
    # Tầng 4: + Lịch sử trả nợ nội bộ TNF (Repayment History)
    stage4_num = stage3_num + ['is_existing_cust', 'repay_so_ky_vay', 'repay_dpd_max', 
                               'repay_dpd_mean', 'repay_so_lan_tre', 'repay_ty_le_tre', 
                               'repay_so_lan_thieu', 'repay_ty_le_thieu']
    stage4_cat = stage3_cat
    
    # Tầng 5: + Bảng lương (Payroll) & Biến liên nguồn (DTI, Độ phóng đại thu nhập)
    stage5_num = stage4_num + ['has_payroll', 'payroll_so_thang', 'payroll_luong_gan_nhat', 
                               'payroll_luong_mean', 'payroll_luong_std', 'dti_tong', 'do_phong_dai_thu_nhap']
    stage5_cat = stage4_cat
    
    return [
        ("1. Chỉ Hồ sơ", stage1_num, stage1_cat),
        ("2. + Khách hàng & Đại lý", stage2_num, stage2_cat),
        ("3. + Credit Bureau", stage3_num, stage3_cat),
        ("4. + Lịch sử trả nợ", stage4_num, stage4_cat),
        ("5. + Bảng lương & Liên nguồn", stage5_num, stage5_cat)
    ]

def train_and_evaluate():
    print("=== ĐANG TẢI DỮ LIỆU ĐÃ TÍCH HỢP ===")
    train_df = pd.read_parquet('data/train_processed.parquet')
    test_df = pd.read_parquet('data/test_processed.parquet')
    
    y = train_df['vo_no'].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    feature_stages = get_feature_sets()
    ablation_results = []
    
    print("\n=== BẮT ĐẦU ABLATION STUDY QUA 5 TẦNG NGUỒN DỮ LIỆU ===")
    
    test_preds_final = np.zeros(len(test_df))
    oof_preds_final = np.zeros(len(train_df))
    
    for stage_idx, (stage_name, num_cols, cat_cols) in enumerate(feature_stages):
        # Tiền xử lý: Imputer -> Scaler cho số, Imputer -> OneHot cho chữ
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ])
        
        # Mô hình Logistic Regression với Regularization L2 để kiểm soát đa cộng tuyến
        model = LogisticRegression(C=0.1, max_iter=1000, random_state=42, solver='lbfgs')
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        oof_preds = np.zeros(len(train_df))
        test_preds = np.zeros(len(test_df))
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(train_df, y)):
            X_train = train_df.iloc[train_idx]
            y_train = y[train_idx]
            X_val = train_df.iloc[val_idx]
            y_val = y[val_idx]
            
            pipeline.fit(X_train, y_train)
            
            # Dự đoán xác suất cho fold validation
            oof_preds[val_idx] = pipeline.predict_proba(X_val)[:, 1]
            # Dự đoán tích lũy cho tập test
            test_preds += pipeline.predict_proba(test_df)[:, 1] / 5.0
            
        auc = roc_auc_score(y, oof_preds)
        ks = calculate_ks(y, oof_preds)
        
        print(f"[{stage_name}] -> AUC: {auc:.4f} | KS: {ks:.4f}")
        ablation_results.append({
            'Tầng dữ liệu': stage_name,
            'Số biến': len(num_cols) + len(cat_cols),
            'AUC (5-Fold CV)': round(auc, 4),
            'KS Statistic': round(ks, 4)
        })
        
        # Nếu là tầng cuối cùng (Full model), lưu lại kết quả dự đoán
        if stage_idx == len(feature_stages) - 1:
            oof_preds_final = oof_preds
            test_preds_final = test_preds

    # -------------------------------------------------------------
    # LƯU KẾT QUẢ ABLATION VÀ FILE DỰ ĐOÁN
    # -------------------------------------------------------------
    ablation_df = pd.DataFrame(ablation_results)
    ablation_df['Mức tăng AUC (ΔAUC)'] = ablation_df['AUC (5-Fold CV)'].diff().fillna(0).round(4)
    ablation_df.to_csv('outputs/ablation_results.csv', index=False)
    print("\n=== KẾT QUẢ SO SÁNH ABLATION STUDY ===")
    print(ablation_df.to_string(index=False))

    # Xuất file dự đoán cho tập Test
    sub = pd.DataFrame({
        'ma_ho_so': test_df['ma_ho_so'],
        'xac_suat_vo_no': test_preds_final
    })
    sub.to_csv('outputs/du_doan.csv', index=False)
    print(f"\n-> Đã xuất thành công file dự đoán: outputs/du_doan.csv (Tổng cộng {len(sub):,} dòng)")
    print(sub.head())

    # -------------------------------------------------------------
    # ĐÁNH GIÁ MA TRẬN NHẦM LẪN TẠI MỘT VÀI NGƯỠNG CUT-OFF
    # -------------------------------------------------------------
    print("\n=== MA TRẬN NHẦM LẪN TẠI CÁC NGƯỠNG CUT-OFF (FULL MODEL) ===")
    thresholds = [0.10, 0.15, 0.20, 0.30]
    for th in thresholds:
        y_pred = (oof_preds_final >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        approval_rate = (y_pred == 0).mean() # Tỷ lệ duyệt (hồ sơ được dự đoán là không vỡ nợ)
        bad_rate_approved = y[y_pred == 0].mean() if approval_rate > 0 else 0 # Tỷ lệ nợ xấu lọt lưới trong nhóm được duyệt
        print(f"Ngưỡng {th:.2f}: TP={tp:,}, FP={fp:,}, FN={fn:,}, TN={tn:,} | Tỷ lệ duyệt: {approval_rate:.1%} | Nợ xấu trong nhóm duyệt: {bad_rate_approved:.2%}")

    # Vẽ biểu đồ Ablation Study
    os.makedirs('report/figures', exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(ablation_df['Tầng dữ liệu'], ablation_df['AUC (5-Fold CV)'], marker='o', color='#1f77b4', linewidth=2.5, markersize=8)
    for i, txt in enumerate(ablation_df['AUC (5-Fold CV)']):
        plt.annotate(f"{txt:.4f}", (ablation_df['Tầng dữ liệu'][i], ablation_df['AUC (5-Fold CV)'][i] + 0.005), 
                     ha='center', fontweight='bold', fontsize=11)
    plt.title('Ablation Study: Mức tăng trưởng AUC khi tích lũy từng nguồn dữ liệu', fontsize=14, pad=15)
    plt.xlabel('Tầng dữ liệu tích lũy', fontsize=12)
    plt.ylabel('ROC-AUC (5-Fold CV)', fontsize=12)
    plt.ylim(ablation_df['AUC (5-Fold CV)'].min() - 0.05, ablation_df['AUC (5-Fold CV)'].max() + 0.05)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('report/figures/ablation_auc_curve.png', dpi=300)
    print("\n-> Đã lưu biểu đồ: report/figures/ablation_auc_curve.png")

if __name__ == '__main__':
    train_and_evaluate()
