# ĐỀ BÀI: CHẤM ĐIỂM TÍN DỤNG TIÊU DÙNG TỪ NHIỀU NGUỒN DỮ LIỆU

---

## 1. Bối cảnh
- **TNF Finance** (công ty giả định) cho vay trả góp tại điểm bán: điện máy, điện thoại và xe máy, thông qua khoảng 300 đại lý ở các tỉnh phía Nam. Khách chọn sản phẩm tại cửa hàng, nhân viên đại lý nhập hồ sơ vay, bộ phận thẩm định tra cứu thông tin và duyệt thủ công.
- Tỷ lệ nợ xấu đang ở mức khoảng **12%**, thẩm định thủ công chậm và không nhất quán.
- Ban giám đốc cần một mô hình chấm điểm ước lượng xác suất vỡ nợ của mỗi hồ sơ ngay khi nộp, và mô hình phải giải thích được $\rightarrow$ Công ty chọn **Logistic Regression**.
- **Thách thức:** Hồ sơ vay chỉ chứa những gì khách tự khai. Thông tin phản ánh đúng rủi ro (thu nhập thực, các khoản nợ ở nơi khác, thói quen trả nợ) nằm rải rác ở nhiều hệ thống khác nhau. Muốn có mô hình tốt, trước hết phải tích hợp được các nguồn này.

---

## 2. Bài toán

| Thành phần | Định nghĩa |
| :--- | :--- |
| **Đơn vị quan sát** | Một hồ sơ vay (`ma_ho_so`) tại thời điểm nộp |
| **Biến mục tiêu** | `vo_no` = 1 nếu khoản vay quá hạn từ 90 ngày trở lên trong 12 tháng đầu, ngược lại = 0 |
| **Mô hình** | Logistic Regression |
| **Nguyên tắc thời gian** | **Chỉ dùng thông tin có trước ngày nộp hồ sơ (`ngay_nop`)** |

---

## 3. Dữ liệu
Dữ liệu gồm 7 file CSV (UTF-8), mô phỏng 6 hệ thống nguồn của công ty. Đơn vị tiền tệ: **VNĐ**.

### 3.1. Tổng quan các file dữ liệu

| File | Hệ thống nguồn | Mỗi dòng là | Số dòng | Khóa nối |
| :--- | :--- | :--- | :--- | :--- |
| `applications_train.csv` | Phần mềm tại đại lý | 1 hồ sơ vay (có nhãn) | 16.046 | `ma_ho_so`, `ma_khach_hang`, `ma_dai_ly` |
| `applications_test.csv` | Phần mềm tại đại lý | 1 hồ sơ vay (không nhãn) | 3.954 | Như trên |
| `customers.csv` | CRM | 1 khách hàng | 20.000 | `ma_khach_hang` |
| `credit_bureau.csv` | Trung tâm thông tin tín dụng (CIC/Bureau) | 1 khoản vay của khách tại tổ chức khác, tra cứu lúc nộp hồ sơ | 20.300 | `ma_ho_so` |
| `repayment_history.csv` | Hệ thống quản lý khoản vay | 1 kỳ thanh toán của khoản vay cũ tại TNF | 179.265 | `ma_khach_hang` |
| `payroll.csv` | Chương trình liên kết trả lương | 1 khách × 1 tháng lương | 259.416 | `ma_khach_hang` |
| `dealers.csv` | Phòng kinh doanh | 1 đại lý | 300 | `ma_dai_ly` |

---

### 3.2. Mô tả chi tiết các cột

#### a. `applications_train.csv` & `applications_test.csv`
- `ma_ho_so`, `ma_khach_hang`, `ma_dai_ly`: Mã định danh hồ sơ, khách hàng, đại lý.
- `ngay_nop`: Ngày nộp hồ sơ (khoảng từ 01/2024 – 06/2025).
- `loai_san_pham`: `dien_may`, `dien_thoai`, `xe_may`.
- `nghe_nghiep`: Nghề nghiệp do khách tự khai.
- `gia_tri_hang`, `tra_truoc`, `so_tien_vay`: Giá trị hàng = `tra_truoc` + `so_tien_vay` (đồng).
- `ky_han_thang`, `lai_suat_nam`: Kỳ hạn vay (tháng) và lãi suất năm.
- `tra_gop_thang`: Số tiền trả góp hàng tháng của khoản vay mới (đồng).
- `thu_nhap_khai_bao`: Thu nhập tháng khách tự khai (đồng).
- `vo_no`: Nhãn mục tiêu (chỉ có trong file train, 1: vỡ nợ, 0: không vỡ nợ).

#### b. `customers.csv`
- `ma_khach_hang`, `ngay_sinh`, `tinh`, `tinh_trang_hon_nhan`, `so_nguoi_phu_thuoc`, `so_nam_cu_tru`, `loai_nha_o`.

#### c. `credit_bureau.csv`
- `ma_ho_so`, `ma_khach_hang`, `to_chuc` (tổ chức cho vay), `loai_vay`, `du_no_hien_tai`, `tra_hang_thang`, `so_lan_qua_han_12t` (số lần quá hạn trong 12 tháng gần nhất), `nhom_no_cao_nhat` (1 = đủ tiêu chuẩn ... 5 = có khả năng mất vốn), `ngay_bao_cao`.
- *Lưu ý:* Hồ sơ không có dòng nào nghĩa là khách không có khoản vay ở nơi khác.

#### d. `repayment_history.csv`
- `ma_khoan_vay_cu`, `ma_khach_hang`, `ky_thu`, `ngay_den_han`, `ngay_thanh_toan`, `so_tien_phai_tra`, `so_tien_da_tra`.
- *Lưu ý:* Toàn bộ các kỳ đều trước năm 2024. Khách không có dòng nào là khách vay lần đầu tại TNF.

#### e. `payroll.csv`
- `ma_khach_hang`, `thang` (YYYY-MM, từ 2023-07 đến 2025-06), `luong_thuc_nhan` (đồng).
- *Lưu ý:* Chỉ khách làm việc tại doanh nghiệp đối tác mới có dữ liệu.

#### f. `dealers.csv`
- `ma_dai_ly`, `tinh`, `loai_dai_ly`, `so_nam_hop_tac`, `quy_mo` (`nho`, `vua`, `lon`).

---

### 3.3. Quan hệ giữa các nguồn & Nguyên tắc tích hợp
- `applications` $\rightarrow$ `customers`: Nhiều–một theo `ma_khach_hang`.
- `applications` $\rightarrow$ `dealers`: Nhiều–một theo `ma_dai_ly`.
- `applications` $\rightarrow$ `credit_bureau`: Một–nhiều theo `ma_ho_so`, cần gộp về **1 dòng/hồ sơ**.
- `applications` $\rightarrow$ `repayment_history`: Một–nhiều theo `ma_khach_hang`, cần gộp về **1 dòng/khách**.
- `applications` $\rightarrow$ `payroll`: Một–nhiều theo `ma_khach_hang`, cần gộp các tháng **trước ngày nộp** (`ngay_nop`).
- **Kết quả tích hợp:** Bảng phân tích với **1 dòng = 1 hồ sơ**, số dòng phải đúng bằng số hồ sơ ban đầu.

---

## 4. Yêu cầu chi tiết

1. **Hiểu nghiệp vụ:**
   - Trình bày đơn vị quan sát, biến mục tiêu, mốc thời gian quan sát.
   - Giải thích vì sao chỉ dùng thông tin trước ngày nộp hồ sơ (chống rò rỉ dữ liệu / Data Leakage).
2. **Khám phá từng nguồn:**
   - Với mỗi file: mỗi dòng đại diện cho gì, khóa nối, quan hệ với bảng hồ sơ, tỷ lệ hồ sơ/khách có dữ liệu ở nguồn đó.
3. **Tích hợp dữ liệu:**
   - Gộp các bảng chi tiết (bureau, lịch sử trả nợ, sao kê lương) về đúng cấp, nối về bảng hồ sơ.
   - Xử lý hợp lý các hồ sơ không có dữ liệu ở một nguồn (khách mới, khách không có sao kê lương...).
4. **Tạo biến liên nguồn (Feature Engineering):**
   - Tối thiểu gồm:
     - **DTI tổng** = $(\text{trả góp mới} + \text{trả nợ ngoài hàng tháng}) / \text{thu nhập ước tính}$
     - **Độ phóng đại thu nhập** = $1 - (\text{thu nhập thực} / \text{thu nhập khai báo})$
   - Khuyến khích đề xuất thêm các biến phái sinh khác.
5. **Huấn luyện mô hình:**
   - Logistic Regression trên tập train (có thể dùng chuẩn hóa, one-hot encoding, regularization: L1/L2/ElasticNet).
6. **Đánh giá mô hình:**
   - Đánh giá bằng **AUC**, **KS** và **Confusion Matrix** tại một vài ngưỡng cut-off.
   - Dự đoán xác suất vỡ nợ cho `applications_test.csv`.
7. **Phân tích đóng góp từng nguồn (Feature Source Importance / Ablation Study):**
   - Huấn luyện lại mô hình khi thêm dần từng nguồn:
     1. Chỉ hồ sơ
     2. $+$ Khách hàng, Đại lý
     3. $+$ Bureau
     4. $+$ Lịch sử trả nợ
     5. $+$ Sao kê lương và biến liên nguồn
   - Lập bảng so sánh và vẽ biểu đồ AUC, rút ra nhận xét chuyên sâu.

---

## 5. Sản phẩm nộp
- **Notebook (`.ipynb`):** Chạy được trơn tru từ đầu đến cuối với dữ liệu gốc.
- **File dự đoán (`du_doan.csv`):** Gồm 2 cột `ma_ho_so`, `xac_suat_vo_no` cho toàn bộ hồ sơ trong `applications_test.csv`.
- **Báo cáo ngắn:** Tối đa 8 trang trình bày quy trình tích hợp, các biến đã tạo, kết quả và bảng đóng góp từng nguồn.

---

## 6. Thang điểm gợi ý

| Hạng mục | Tỷ trọng |
| :--- | :--- |
| Hiểu nghiệp vụ và mô tả dữ liệu | 10% |
| Tích hợp dữ liệu đúng | 30% |
| Biến liên nguồn | 20% |
| Mô hình, đánh giá và AUC trên tập test ẩn | 20% |
| Phân tích đóng góp từng nguồn | 20% |
