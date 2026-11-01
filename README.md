# Tạo ST-VQA suy thoái full trên UIT HPC

Một lệnh **`python run_all.py`** tự tạo `.sh`, submit một job Slurm, chuẩn bị môi trường, xuất clean từ Parquet và tạo toàn bộ ảnh suy thoái. Dùng lại thuật toán và cơ chế chạy của repo TextVQA đã hoàn tất trên HPC.

## 1. Phạm vi của chữ full

Nguồn là [vikhyatk/st-vqa](https://huggingface.co/datasets/vikhyatk/st-vqa), revision:

```text
cbf984aaf398e03a4d9134e6c6c2fd347168e08f
```

Nguồn này có **18.921 dòng ảnh / 26.074 câu hỏi**, chỉ có split **train**. Full nghĩa là xử lý toàn bộ nguồn Hugging Face này, không phải toàn bộ train/val/test chính thức của ST-VQA. Không tự tạo hoặc gắn nhãn val/test.

- Giữ mọi dòng ảnh và tất cả câu hỏi, đáp án có sẵn.
- Không loại ảnh trùng byte để giảm số main. Số dòng trùng được ghi trong provenance; phiên bản đã xuất trước đây có 72 dòng trùng byte.
- ID giống cách xuất nguồn trước đây: `stvqa_train_000000`, `stvqa_train_000001`, ... theo tên shard đã sắp xếp và thứ tự dòng. Question ID là `<image_id>_q00`, `q01`, ...
- Mỗi dòng ảnh tạo **một main PNG realistic_mix L2**, phối hợp ngẫu nhiên 2–3 phép suy thoái như notebook cũ.

Đây là nguồn ảnh/câu hỏi ST-VQA, không yêu cầu JSON TextVQA hoặc dataset suy thoái cũ.

## 2. Vị trí nguồn và output đã cấu hình

```text
/datastore/dungmt/textvqa/data/benchmark_v1/raw/stvqa_clean/
└── data/
    ├── train-00000-of-00006.parquet
    ├── train-00001-of-00006.parquet
    ├── train-00002-of-00006.parquet
    ├── train-00003-of-00006.parquet
    ├── train-00004-of-00006.parquet
    └── train-00005-of-00006.parquet
```

Code tìm các shard ngay dưới `source_root` hoặc `source_root/data`, không quét toàn bộ datastore. Tên và kích thước ở ảnh server đã khớp metadata HF. Sáu SHA-256 gốc được pin trong `configs/hf_source_manifest.json` từ Hugging Face LFS metadata ở revision trên. Hash file thật chỉ được kiểm tra trong allocation, không phải trên login node.

Output:

```text
/datastore/dungmt/textvqa/data/benchmark_v1/raw/full_stvqa_degraded/
```

Không sửa raw Parquet, không tải lại dataset, không cần HF/Kaggle token. Code xuất nguyên byte clean vào **`full_stvqa_degraded/source_clean/images/`** để dùng paired clean/degraded sau này, kể cả MPO; không lưu clean trong một thư mục cache dễ bị xóa.

## 3. Lấy repo và khởi chạy

Trên server UIT, copy nguyên khối sau:

```bash
mkdir -p /datastore/dungmt/textvqa/Tuan
cd /datastore/dungmt/textvqa/Tuan
git clone https://github.com/ncat218/Degraded-STVQA-Image-Generation.git
cd Degraded-STVQA-Image-Generation
python run_all.py
```

Nếu đã clone và không sửa cục bộ:

```bash
cd /datastore/dungmt/textvqa/Tuan/Degraded-STVQA-Image-Generation
git pull --ff-only && python run_all.py
```

Đó là **một lệnh tính toán** cho toàn bộ quá trình. Không chạy riêng export, preflight, calibration, review, freeze hoặc main. Sau khi submit thành công, chương trình in job ID và lệnh `tail -f`; có thể đóng SSH, không submit thêm chỉ vì đã đóng cửa sổ SSH.

Python 3.11 là lựa chọn ưu tiên; hỗ trợ môi trường thư viện Python 3.10–3.13. Nếu chỉ có `python3`, thay lệnh khởi chạy bằng `python3 run_all.py`. Cần Internet trong lần setup đầu để pip cài các thư viện hoặc một cache/mirror đã chuẩn bị. Không cần CUDA toolkit, PyTorch, mô hình OCR/VQA hay checkpoint.

Xem cấu hình mà không submit hoặc đọc ảnh:

```bash
python run_all.py --plan-only
```

Các đường dẫn nằm trong `configs/full_stvqa.json`; không cần đổi với layout server đã gửi. Có thể clone repo vào chỗ khác trong datastore, đường nguồn/output không đổi.

## 4. Slurm và tài nguyên

| Tham số | Mặc định |
|---|---|
| Account | `uit` |
| Partition | `normal` |
| QOS | `mig2-35` |
| GPU GRES | `gpu:2g.35gb:1` |
| CPU / worker | **8 / 8** |
| RAM hệ thống | **40 GB** |
| Walltime | **24 giờ** |

Đây là cấu hình job TextVQA 8108 đã chạy thành công. Không đặt 12/16 CPU chỉ theo mức tối đa partition: yêu cầu 12 CPU trước đó đã bị `QOSMaxCpuPerJobLimit`. Không tự chuyển về `small` đang gặp vấn đề cấp tài nguyên.

Code suy thoái gốc chạy **CPU OpenCV/NumPy**; GPU MIG được cấp theo mẫu allocation của trường nhưng chưa dùng cho biến đổi ảnh. Không tuyên bố dùng H200 để tăng tốc. Việc queue cho phép tác vụ CPU này vẫn phụ thuộc chính sách UIT. Có thể pending vì Priority/Resources; 24 giờ là giới hạn xin chạy, không phải thời gian hoàn tất dự đoán.

Trên login node chỉ đọc cấu hình và kiểm tra tên/kích thước 6 file, rồi gọi `sbatch`. Mọi hash, Parquet decode, xử lý ảnh và xuất manifest ở compute node. Nếu thiếu `sbatch` và allocation thì chương trình dừng; không chạy ảnh trực tiếp trên login.

Workspace, môi trường và log riêng của ST-VQA:

```text
/datastore/dungmt/_workspace_full_stvqa_degraded/
├── venv/
├── pip_cache/
└── logs/                 # .sh, snapshot cấu hình, .out, .err
```

Không ảnh hưởng workspace/output TextVQA đã hoàn tất. Không ghi dataset lớn trong Home.

## 5. Các bước tự động trong cùng job

1. Kiểm tra đúng bộ shard, kích thước, SHA-256, schema `image` và `qas`, tổng số dòng.
2. Đọc Parquet theo batch nhỏ 16 dòng. Không tải toàn bộ shard 500 MB vào pandas.
3. Xuất nguyên byte từng ảnh, kiểm tra giải mã frame 0, kích thước và hash. Nếu đã xuất hợp lệ thì tái sử dụng; log mỗi 1.000 ảnh. Giữ mọi QA và số đáp án thực tế, không pad thành 10 đáp án như TextVQA.
4. Khóa source manifests/provenance và policy. Không resample/bỏ ảnh khỏi main.
5. Chọn cố định 300 ảnh nguồn cho kiểm tra calibration, 120 ảnh trong đó cho lỗi đơn; chọn 400 ảnh cho ablation. Tất cả vẫn có main: calibration không phải holdout đánh giá độc lập.
6. Preflight 25 ảnh để ước lượng tài nguyên; kiểm tra dung lượng trống và trần output 100 GB, biên 25%.
7. Calibration L1/L2/L3: **3.420 ảnh phụ**.
8. Xuất gallery/template và nhãn tự động 70/30; khóa cấu hình.
9. Frozen preflight rồi tạo **18.921 main PNG realistic_mix L2**.
10. Tạo **2.800 ảnh ablation**: 400 ảnh × 7 lỗi đơn L2. Kiểm tra lại disk sau main.
11. Xác minh hash mọi output; xuất manifest câu hỏi, metadata JSONL và báo cáo complete.

CSV nhãn dùng `EASY` và `DIFFICULT_BUT_ANSWERABLE`, khoảng 70/30 (615 dòng mặc định: 431/184). **Đây là nhãn mô phỏng ngẫu nhiên, không có human review thực tế.** CSV đi kèm provenance; frozen config cũng ghi `human_review_performed=False`. Nhãn không chọn/loại ảnh hoặc chứng minh answerability. Không phải thao tác thủ công giữa các bước.

## 6. Thuật toán được giữ lại

Giữ nguyên cell code toán tử và `DEFAULT_PARAMS` từ notebook Final cũ; hash cell:

```text
86e80ed9e98d4607852dfdbe0ff9f5381aba1c81849251b009cb54834333c12e
```

Gồm motion/camera shake PSF đường thẳng hoặc random walk, defocus kernel đĩa/ellipse, low light + Poisson/Gaussian noise, overexposure/glare Gaussian, downsample AREA/upsample CUBIC, perspective homography có guard biên 5% và JPEG encode/decode theo quality.

Mix L2 chọn 2–3 phép, không cùng lúc motion + defocus, không cùng lúc low light + overexposure; giữ nguyên thứ tự áp dụng gốc. Seed được suy từ seed chung + source manifest hash + config hash + split + image ID + condition + level, độc lập thứ tự worker.

HF source không có OCR boxes nên dùng nhánh `image_relative_fallback` như bản ST-VQA cũ. Không thêm OCR, không giả kích thước chữ. MPO/multi-frame: **clean giữ nguyên byte, suy thoái chỉ frame 0**; metadata lưu `source_format`, `source_frame_index`, `source_frames`. Không EXIF-transpose và không re-encode clean sang JPEG. PNG degraded RGB giữ nguyên kích thước frame clean.

Giữ thuật toán không có nghĩa pixel phải trùng dataset 5k cũ: manifest/config của bản full thay đổi nên seed mỗi ảnh thay đổi. Với cùng nguồn/code/config/thư viện thì resume và số worker khác nhau phải giữ cùng output.

## 7. Cây output

```text
full_stvqa_degraded/
├── source_clean/
│   ├── images/stvqa_train_000000.<jpg|png|mpo|...>
│   ├── image_manifest.csv
│   └── question_manifest.csv
├── main/
│   ├── images/realistic_mix/L2/train/
│   │   ├── stvqa_train_000000.png
│   │   ├── stvqa_train_000000.png.json
│   │   └── ...                            # 18.921 PNG + sidecar
│   ├── final_main_manifest.csv            # 26.074 QA rows
│   ├── question_manifest.csv
│   ├── image_manifest.csv                 # 18.921 image rows
│   ├── manifests/train.csv
│   ├── diagnostics_summary.json
│   └── failures.json
├── calibration/
│   ├── images/<condition>/<L1|L2|L3>/train/...
│   ├── calibration_manifest.csv
│   ├── image_manifest.csv
│   └── manual_review/
│       ├── review_gallery.html
│       ├── previews/...
│       ├── review_samples.csv
│       ├── reviewed_samples.csv
│       └── reviewed_samples.random_label_provenance.json
├── ablation/
│   ├── images/<condition>/L2/train/...
│   ├── ablation_manifest.csv              # 400 source rows
│   ├── image_manifest.csv                 # 2.800 output rows
│   └── question_manifest.csv
├── config/
│   ├── draft_degradation_config.json
│   └── frozen_degradation_config.json
├── metadata/
│   ├── hf_source_manifest.json
│   ├── source_image_manifest.csv
│   ├── locked_eval_manifest.csv           # tên tương thích; full HF train QA
│   ├── source_provenance.json
│   ├── split_policy.json
│   └── degradation_params.jsonl
├── preflight/
├── environment.json
├── generation_summary.json
└── pipeline_failure.json
```

Manifest chính có dataset, split, image/source/question ID, câu hỏi, đáp án, số đáp án, index nguồn/QA, đường dẫn clean/degraded, kích thước, format/frame nguồn, condition/level, seed, params, source/output/config hash. Ảnh có nhiều câu hỏi dùng cùng PNG; JSON sidecar không phải ảnh bổ sung. `params` cho biết từng lỗi thực sự áp dụng trong realistic_mix.

Giữ `source_clean/` nếu dùng paired comparison. Nếu copy dataset sang vị trí khác, đường dẫn tuyệt đối trong manifest phải được ánh xạ bằng ID/hash tương ứng, không tùy ý đổi ID. Nếu tự chia tập train/val/test cho thí nghiệm sau này, cần nhóm theo cả ID và hash nguồn để ảnh trùng byte/các biến thể không lọt sang split khác. Đây không tự trở thành split chính thức ST-VQA.

## 8. Xem tiến độ và xác nhận hoàn tất

Sau khi submit, code in đường dẫn log cụ thể. Dùng:

```bash
squeue -u "$USER"
ls -lt /datastore/dungmt/_workspace_full_stvqa_degraded/logs/
tail -n 100 -f /datastore/dungmt/_workspace_full_stvqa_degraded/logs/<ten_file>.out
tail -n 100 /datastore/dungmt/_workspace_full_stvqa_degraded/logs/<ten_file>.err
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed
```

Thay placeholder bằng tên/job ID thực. Ctrl+C chỉ thoát tail, không hủy job. Trong giai đoạn tạo ảnh, log mỗi 100 output. Trước đó có log verify shard và export clean, không nhầm với tạo degraded main.

Hoàn tất khi Slurm `COMPLETED`, `ExitCode=0:0` và báo cáo có:

```text
status: complete
main_images: 18921
main_questions: 26074
main outputs: 18921 failures: 0
calibration outputs: 3420 failures: 0
ablation outputs: 2800 failures: 0
READY FOR DOWNSTREAM ST-VQA / VQA EVALUATION: main, ablation
```

Kiểm tra báo cáo nhanh:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p=Path('/datastore/dungmt/textvqa/data/benchmark_v1/raw/full_stvqa_degraded/generation_summary.json')
if not p.is_file():
    print('Chưa có báo cáo hoàn tất')
else:
    d=json.loads(p.read_text())
    print(d.get('status'),d.get('main_images'),d.get('main_questions'))
    for name,s in d.get('stages',{}).items():
        print(name,s['output_count'],'lỗi:',s['failure_count'])
PY
```

## 9. Resume và lỗi

Chạy lại cùng `python run_all.py` sau khi job lỗi/hết giờ sẽ verify lại 6 shard, clean đã xuất và các PNG/sidecar. Ảnh hợp lệ được tái sử dụng; PNG mồ côi sau ngắt có thể tạo lại xác định. Hai job không được cùng ghi một output; có khóa OS.

Không rerun để theo dõi một job đang chạy. Tôn trọng giới hạn submit của trường. Có thể tăng thời gian/trần disk hoặc đổi worker trong mức allocation để resume; không đổi seed, tham số, nguồn, code hoặc output root giữa lượt dở dang. Không `git pull` đổi thuật toán khi một job đang chạy; các code hashes được khóa.

- Shard thiếu/thừa, sai size hoặc SHA: dừng, không tự tải nguồn khác hay bỏ kiểm tra.
- Clean đã xuất bị đổi byte: dừng, không ghi đè lặng lẽ.
- Source, config hoặc metadata khóa đổi: dùng output mới sau khi xác minh nguyên nhân.
- PNG/sidecar không khớp: dừng; lỗi ở `pipeline_failure.json` hoặc `<stage>/failure.json`.
- Disk limit mặc định 100 GB là trần cho toàn output (gồm clean đã xuất), không bảo đảm quota trường có sẵn; code kiểm tra dung lượng thực và biên dự phòng.

## 10. Kiểm chứng và bố trí repo

```text
run_all.py                        # file duy nhất người dùng khởi chạy
configs/full_stvqa.json            # đường dẫn + Slurm
configs/hf_source_manifest.json    # pin shard names/sizes/SHA tại revision HF
requirements.txt                  # thư viện xử lý ảnh + PyArrow đã pin
src/stvqa_degradation/             # controller helper, importer, pipeline, toán tử
tests/test_stvqa_full.py           # fixture Parquet/MPO/QA và resume
operator_provenance.json
VERIFICATION.md
```

Kiểm thử dành cho người phát triển có môi trường requirements:

```bash
python -m unittest discover -s tests -v
```

Đã kiểm thử cục bộ trên sáu shard Parquet nhỏ có MPO thật, byte duplicate, nhiều QA/answer count, tạo PNG, frame 0, resume và multi-process. Chưa chạy bản full ST-VQA thật trên UIT; không có benchmark thời gian trước. Giới hạn kiểm chứng ghi trong `VERIFICATION.md`. Không upload raw data/output/venv hoặc thông tin đăng nhập vào repo public.
