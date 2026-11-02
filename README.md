# Xu_ly_anh — Pic2Word (CVPR 2023) bản mini

Thử nghiệm **zero-shot composed image retrieval** (ZS-CIR) theo bài
*Saito et al., "Pic2Word: Mapping Pictures to Words for Zero-shot Composed Image Retrieval", CVPR 2023*
([code gốc của Google Research](https://github.com/google-research/composed_image_retrieval)) — phục vụ học phần Xử lý ảnh.

## Pic2Word là gì
- Query = **ảnh + câu chữ** (ví dụ: ảnh chiếc ghế + "a sketch of *") → tìm ảnh đúng ý trong kho ảnh, **không cần dữ liệu triplet có nhãn**.
- Cách làm: CLIP đóng băng; train một **mapping network** (MLP) biến image embedding thành một **pseudo-word token** chèn vào câu text. Train bằng InfoNCE trên các cặp ảnh–caption (không cần nhãn).

## Bản mini trong repo này
- `pic2word_kaggle.ipynb` — notebook chạy **Kaggle GPU** (push headless bằng `kaggle kernels push`, không cần mở web).
  - Dữ liệu: Kaggle dataset `aladdinpersson/flickr8kimagescaptions` (Flickr8k, 8000 ảnh + captions.txt).
  - CLIP ViT-B/32 đóng băng + mapper MLP 2 lớp; train 2000 ảnh × 2 epoch (~10 phút GPU).
  - Eval self-retrieval Recall@1/5 trên 200 ảnh held-out, có baseline so sánh (raw image embedding không qua mapper).
  - Demo grid ảnh query → top-4 kết quả.
- `build_pic2word_nb.py` — script sinh lại notebook trên.
- `sample_data/` — dữ liệu mẫu mặc định của môi trường notebook (giữ nguyên).

## Giới hạn (nói thẳng)
Bản mini chỉ dạy mapper khớp **ảnh gốc** nên chỉ đo được self-retrieval; các template ZS-CIR thật ("a sketch of *", đổi thuộc tính) cần train quy mô CC3M (~3M cặp) như bài gốc mới tổng quát. Đây là demo cơ chế để hiểu phương pháp, không phải tái lập kết quả SOTA.
