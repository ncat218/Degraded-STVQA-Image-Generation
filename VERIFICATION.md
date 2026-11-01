# Kiểm chứng ST-VQA full trước bàn giao

Ngày 2026-11-01. Test cục bộ trên Windows, Python 3.11, thư viện pin ở requirements.

`python -m unittest discover -s tests -v`: **13 tests passed**.

Đã kiểm tra:

- Code cell và DEFAULT_PARAMS của toán tử giữ đúng hash bản Final cũ.
- Determinism, shape/dtype và quy tắc phối hợp lỗi L2.
- Cấu hình normal 8 CPU, QOS mig2-35; controller gọi sbatch một lần (mock), không fallback tính toán trên login.
- Tên/size shard được kiểm tra trước submit; thiếu/thừa shard bị từ chối.
- Đọc sáu shard Parquet nhỏ theo batch: 40 ảnh nguồn, 54 câu hỏi, 117 calibration, 28 ablation theo cấu hình fixture.
- Tất cả ảnh, QA và số reference answers được giữ; không tự tạo val/test; byte duplicate không bị bỏ khỏi main.
- Clean bytes khớp hoàn toàn với Parquet; có MPO hai frame thật; output khớp phép suy thoái trên frame 0.
- Resume không ghi lại clean hợp lệ, giữ nguyên PNG và manifest.
- Hai process thực sự tạo lại PNG giống bản tuần tự.
- Phục hồi sidecar thiếu, chặn sidecar sai, raw Parquet đổi byte hoặc clean đã xuất bị sửa.
- Nhãn review mô phỏng 70/30 được ghi rõ `human_review_performed=False`.

Đã lấy metadata Hugging Face API tại revision cbf984aaf398e03a4d9134e6c6c2fd347168e08f để pin sáu tên, kích thước và LFS SHA-256. Kích thước khớp screenshot server; chưa truy cập trực tiếp file trên UIT hoặc tải full Parquet vào máy cục bộ. Kiểm tra SHA/schema/count thực tế sẽ diễn ra trong job trước khi tạo suy thoái.

Chưa submit job ST-VQA lên UIT, chưa đo thời gian/dung lượng full thật và chưa khẳng định CI Linux đã chạy. Cấu hình Slurm kế thừa job TextVQA 8108 mà người dùng xác nhận COMPLETED trong 01:12:00; đó không phải benchmark ST-VQA. Workflow GitHub Actions riêng chạy tests trên Ubuntu sau push.
