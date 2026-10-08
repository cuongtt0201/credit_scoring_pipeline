# KẾ HOẠCH TRIỂN KHAI DỰ ÁN (WORK TREE & ROADMAP)
**Dự án:** Chấm điểm tín dụng tiêu dùng từ nhiều nguồn dữ liệu (Credit Scoring - TNF Finance)

---

## 1. Cấu trúc thư mục dự án (Project Work Tree)

Dưới đây là cây thư mục đề xuất chuẩn chỉnh cho dự án Data Science & Credit Scoring, phân định rõ ràng giữa dữ liệu, mã nguồn phân tích/mô hình hóa, kết quả dự đoán và báo cáo:

```text
assignment/
│
├── DE_BAI.md                   # File đề bài gốc để tham chiếu
├── WORK_PLAN.md                # Kế hoạch chi tiết, checklist & tiến độ triển khai
│
├── data/                       # Chứa 7 file dữ liệu đầu vào (CSV)
│   ├── applications_train.csv  # 16.046 dòng (có nhãn vo_no)
│   ├── applications_test.csv   # 3.954 dòng (tập test cần dự đoán)
│   ├── customers.csv           # Thông tin định danh & nhân khẩu học
│   ├── credit_bureau.csv       # Lịch sử nợ tại các tổ chức tín dụng khác
│   ├── repayment_history.csv   # Lịch sử trả nợ các khoản vay cũ tại TNF
│   ├── payroll.csv             # Sao kê lương liên kết qua doanh nghiệp
│   └── dealers.csv             # Thông tin 300 đại lý phân phối
│
├── notebooks/                  # Jupyter Notebook thực thi
│   └── credit_scoring_pipeline.ipynb  # Notebook chính (chạy trơn tru từ A -> Z)
│
├── outputs/                    # Sản phẩm đầu ra
│   ├── du_doan.csv             # File dự đoán (ma_ho_so, xac_suat_vo_no)
│   └── ablation_results.csv    # Kết quả so sánh AUC qua từng nguồn dữ liệu
│
└── report/                     # Báo cáo nộp bài (tối đa 8 trang)
    ├── report.md               # Nội dung nháp báo cáo chi tiết
    └── figures/                # Biểu đồ phục vụ báo cáo (ROC curve, KS plot, Ablation chart)
```

---

## 2. Phân rã công việc chi tiết (Task Breakdown theo yêu cầu nộp)

### Phase 1: Hiểu nghiệp vụ & Khám phá dữ liệu (EDA) — [10% số điểm]
- [ ] **1.1. Phân tích nghiệp vụ:**
  - Làm rõ đơn vị quan sát (`ma_ho_so`), biến mục tiêu (`vo_no`), bản chất bài toán nhị phân.
  - Phân tích nguyên tắc rò rỉ dữ liệu (Point-in-time validation): Vì sao **chỉ** được lấy dữ liệu $\le$ `ngay_nop`?
- [ ] **1.2. Khám phá 7 file dữ liệu (EDA từng nguồn):**
  - Kiểm tra số lượng dòng, kiểu dữ liệu, tỷ lệ missing/null.
  - Phân tích tỷ lệ bao phủ (coverage rate): Bao nhiêu % hồ sơ có trong Bureau, Repayment, Payroll, Dealers, Customers.
  - Thống kê tỷ lệ nợ xấu nền (base rate ~12%) trên tập train.

---

### Phase 2: Tích hợp dữ liệu (Data Integration & Aggregation) — [30% số điểm]
*Mục tiêu: Đưa tất cả dữ liệu từ quan hệ 1-N về 1-1 theo `ma_ho_so`, bảng kết quả giữ nguyên đúng 16.046 dòng (train) và 3.954 dòng (test).*

- [ ] **2.1. Tích hợp `customers` & `dealers`:**
  - Nối bảng nhiều-một theo `ma_khach_hang` và `ma_dai_ly`.
- [ ] **2.2. Gộp và tích hợp `credit_bureau` (1-N theo `ma_ho_so`):**
  - Tổng dư nợ hiện tại, tổng nghĩa vụ trả nợ ngoài hàng tháng (`sum(tra_hang_thang)`).
  - Nhóm nợ xấu nhất (`max(nhom_no_cao_nhat)`), số lần quá hạn tối đa/tổng (`sum/max(so_lan_qua_han_12t)`).
  - Số lượng khoản vay ngoài.
  - *Xử lý missing:* Điền 0 (cho dư nợ, số lần quá hạn) + cờ báo hiệu `has_bureau_record = 0/1`.
- [ ] **2.3. Gộp và tích hợp `repayment_history` (1-N theo `ma_khach_hang`):**
  - Kiểm tra các kỳ trả nợ (đều trước 2024 $\le$ `ngay_nop`).
  - Tạo chỉ số hành vi: Số ngày trễ hạn trung bình / trễ tối đa (`DPD = ngay_thanh_toan - ngay_den_han`).
  - Tỷ lệ trả thiếu tiền (`so_tien_da_tra < so_tien_phai_tra`), số khoản vay cũ đã từng hoàn thành.
  - *Xử lý missing:* Gán cờ `is_new_customer = 1` (khách hàng mới vay TNF lần đầu).
- [ ] **2.4. Gộp và tích hợp `payroll` (1-N theo `ma_khach_hang` có điều kiện thời gian):**
  - **Chặn thời gian nghiêm ngặt:** Chỉ lấy các tháng lương có `thang <= ngay_nop`.
  - Tính: Lương trung bình 3 tháng gần nhất, 6 tháng gần nhất, độ biến động lương (std).
  - *Xử lý missing:* Cờ `has_payroll = 0/1`, ước tính thu nhập dựa trên lương thực tế hoặc khai báo.

---

### Phase 3: Kỹ nghệ đặc trưng liên nguồn (Cross-source Feature Engineering) — [20% số điểm]
- [ ] **3.1. Các biến bắt buộc theo đề bài:**
  - **DTI tổng (Total Debt-to-Income):**
    $$\text{DTI} = \frac{\text{tra\_gop\_thang} + \text{tra\_hang\_thang\_ngoai}}{\text{thu\_nhap\_uoc\_tinh}}$$
  - **Độ phóng đại thu nhập (Income Exaggeration):**
    $$\text{Exaggeration} = 1 - \frac{\text{thu\_nhap\_thuc\_nhan}}{\text{thu\_nhap\_khai\_bao}}$$
- [ ] **3.2. Đề xuất các biến phái sinh nâng cao (tạo điểm cộng):**
  - Tỷ lệ trả trước: `tra_truoc / gia_tri_hang` (LTV - Loan to Value).
  - Gánh nặng trả góp riêng tại TNF: `tra_gop_thang / thu_nhap`.
  - Tuổi khách hàng: `(ngay_nop - ngay_sinh) / 365`.
  - Tỷ lệ thu nhập trên số người phụ thuộc: `thu_nhap / (so_nguoi_phu_thuoc + 1)`.
  - Rủi ro đại lý: Tỷ lệ nợ xấu lịch sử theo đại lý / theo loại hàng / theo tỉnh thành.

---

### Phase 4: Huấn luyện & Đánh giá mô hình Logistic Regression — [20% số điểm]
- [ ] **4.1. Tiền xử lý dữ liệu chuẩn cho Logistic Regression:**
  - Xử lý Outlier (Winsorization/Clipping) và Missing values (Imputation).
  - Mã hóa biến phân loại (One-Hot Encoding hoặc WoE / Weight of Evidence).
  - Chuẩn hóa thang đo (StandardScaler / RobustScaler) vì Logistic Regression rất nhạy cảm với scale.
  - Phân chia tập validation (Stratified K-Fold hoặc Time-based Split).
- [ ] **4.2. Huấn luyện mô hình:**
  - Huấn luyện Logistic Regression với Regularization (L1 Lasso để loại bỏ biến nhiễu, L2 Ridge, hoặc ElasticNet).
  - Tinh chỉnh siêu tham số ($C$, solver, penalty, class_weight="balanced").
- [ ] **4.3. Đánh giá toàn diện:**
  - Tính chỉ số **ROC-AUC** (trên Validation set).
  - Tính chỉ số **KS statistic** (Kolmogorov-Smirnov) và vẽ biểu đồ tích lũy Bad/Good.
  - Phân tích **Confusion Matrix** tại các ngưỡng cut-off khác nhau (0.1, 0.15, 0.2, 0.3...) cân đối giữa tỷ lệ duyệt vay và tỷ lệ rủi ro nợ xấu.
- [ ] **4.4. Xuất file kết quả dự đoán:**
  - Dự đoán xác suất cho `applications_test.csv`.
  - Xuất ra file `outputs/du_doan.csv` đúng định dạng (`ma_ho_so`, `xac_suat_vo_no`).

---

### Phase 5: Phân tích đóng góp từng nguồn dữ liệu (Ablation Study) — [20% số điểm]
- [ ] **5.1. Huấn luyện lần lượt 5 mô hình theo từng tầng dữ liệu tích lũy:**
  - **Mô hình 1:** Chỉ dữ liệu hồ sơ đăng ký (`applications`).
  - **Mô hình 2:** Thêm thông tin Khách hàng (`customers`) + Đại lý (`dealers`).
  - **Mô hình 3:** Thêm thông tin Trung tâm tín dụng (`credit_bureau`).
  - **Mô hình 4:** Thêm Lịch sử trả nợ nội bộ (`repayment_history`).
  - **Mô hình 5:** Thêm Sao kê lương (`payroll`) + Biến liên nguồn (DTI, Exaggeration,...).
- [ ] **5.2. Lập bảng so sánh & biểu đồ trực quan hóa:**
  - Bảng so sánh AUC, KS tăng thêm qua từng bước ($\Delta \text{AUC}$).
  - Vẽ biểu đồ đường / cột thể hiện sự nhảy vọt của AUC khi thêm từng nguồn.
  - Đưa ra nhận định: Nguồn nào mang lại giá trị phân loại (Information Value) cao nhất? Vì sao?

---

### Phase 6: Hoàn thiện sản phẩm nộp
- [ ] **6.1. Notebook sạch đẹp:**
  - Tổ chức code mạch lạc, có markdown giải thích từng bước, chạy lại Run All từ cell đầu tới cuối không lỗi.
- [ ] **6.2. File `du_doan.csv`:**
  - Kiểm tra độ dài đúng 3.954 dòng, không chứa giá trị NaN/null, xác suất nằm trọn trong khoảng $[0, 1]$.
- [ ] **6.3. Báo cáo (Tối đa 8 trang):**
  - Viết báo cáo hoàn chỉnh gồm: Tóm tắt bài toán $\rightarrow$ Sơ đồ tích hợp $\rightarrow$ Feature Engineering $\rightarrow$ Hiệu năng mô hình & Diễn giải hệ số $\rightarrow$ Kết quả Ablation Study $\rightarrow$ Đề xuất nghiệp vụ cho TNF Finance.
  - *Ghi chú (Data Storytelling & Vận hành ở phần kết luận):*
    - **Data Story:** Không chỉ dừng lại ở con số % dự đoán, mà làm nổi bật câu chuyện thực tế: Khách gian lận khai khống thu nhập ra sao, nguồn dữ liệu nào "đáng đồng tiền bát gạo" nhất (ROI của dữ liệu Bureau/Payroll).
    - **Insight vận hành & Trade-off:** Chiến lược cắt ngưỡng duyệt (Cut-off Threshold) cân bằng giữa Tỷ lệ duyệt (Approval Rate) và Tỷ lệ nợ xấu (Bad Rate).
    - **Bảo trì & Cải tiến:** Nhận ra hạn chế (nhóm khách hàng không có payroll/lương liên kết) và đề xuất hướng giám sát độ ổn định dữ liệu (PSI) & giải pháp thay thế.
