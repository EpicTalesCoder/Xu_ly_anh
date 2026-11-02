# -*- coding: utf-8 -*-
"""Sinh pic2word_kaggle.ipynb — Pic2Word (CVPR 2023) bản mini chạy Kaggle GPU.

Ý tưởng: CLIP đóng băng + mapping network (MLP) biến image embedding ->
pseudo-word token ở vị trí 1 của text input; train InfoNCE trên Flickr8k
(1 caption/ảnh, pseudo-label smoothing 0.2); eval self-retrieval trên ảnh
held-out + demo truy xuất. Chạy headless qua `kaggle kernels push`.

Ghi chú kỹ thuật (rút từ các lần chạy lỗi trên Kaggle):
- KHÔNG gọi clip.text_model(inputs_embeds=...) — bản transformers mới đòi
  input_ids và báo "multiple values for inputs_embeds". Thay vào đó dùng
  forward hook trên token_embedding để chèn pseudo-token ở vị trí 1, rồi gọi
  đường chuẩn text_model(input_ids, attention_mask).
- KHÔNG dùng clip.get_image_features(**inputs) — trả object khi gói mới;
  dùng vision_model + visual_projection thủ công.
- Mount dataset Kaggle: captions.txt và Images có thể KHÔNG nằm cạnh nhau
  (đường dẫn lồng /kaggle/input/datasets/...) — tìm thư mục ảnh bằng glob.
"""
import json, pathlib


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


cells = [
md("""# Pic2Word (CVPR 2023) — bản mini chạy Kaggle GPU

Bài gốc: **Saito et al., "Pic2Word: Mapping Pictures to Words for Zero-shot Composed Image Retrieval", CVPR 2023** ([code gốc](https://github.com/google-research/composed_image_retrieval)).

Ý tưởng: train một **mapping network** biến CLIP image embedding → "pseudo-word" token trong không gian text, sao cho câu chứa pseudo-word có embedding khớp chính ảnh đó. Từ đó truy vấn *ảnh + chữ* (ZS-CIR) không cần dữ liệu triplet.

Bản mini này (phục vụ học phần Xử lý ảnh):
- CLIP ViT-B/32 **đóng băng**, mapper = MLP 2 lớp (512→512).
- Train trên **2000 ảnh Flickr8k** (1 caption/ảnh, pseudo-label smoothing 0.2 như bài gốc), 2 epoch ≈ 10 phút GPU.
- Eval **self-retrieval** trên 200 ảnh held-out (mapper vs nhét raw image embedding), demo truy xuất.
- Kernel chạy headless: Hermes push bằng `kaggle kernels push`, Kaggle tự chạy hết các cell rồi tắt VM."""),

code("""import random, numpy as np, torch

CFG = dict(
    MODEL="openai/clip-vit-base-patch32",
    MAX_TRAIN_IMAGES=2000,
    N_GALLERY=200,
    EPOCHS=2,
    BATCH=64,
    LR=1e-4,
    SMOOTH_P=0.2,      # pseudo-label smoothing: thay caption bằng caption của ảnh khác
    SEED=42,
    TEMPLATE="A photo of .",   # dấu "." (vị trí 1 sau BOS) bị thay bằng pseudo-token
)
random.seed(CFG["SEED"]); np.random.seed(CFG["SEED"]); torch.manual_seed(CFG["SEED"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| torch", torch.__version__)"""),

code("""!pip install -q "transformers>=4.35"
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer

clip = CLIPModel.from_pretrained(CFG["MODEL"]).to(DEVICE).eval()
for p in clip.parameters():
    p.requires_grad_(False)
tok = AutoTokenizer.from_pretrained(CFG["MODEL"])
proc = CLIPProcessor.from_pretrained(CFG["MODEL"])
D = clip.config.projection_dim
print("CLIP ok | projection_dim =", D)"""),

md("""## 1. Dữ liệu Flickr8k (dataset gắn kèm kernel)"""),

code("""import glob, os
import pandas as pd

caps = glob.glob("/kaggle/input/**/captions.txt", recursive=True)
assert caps, "KHÔNG thấy captions.txt — kiểm tra dataset_sources trong kernel-metadata.json"
CAP = caps[0]
print("captions:", CAP)

rows = pd.read_csv(CAP)
if not {"image", "caption"} <= set(rows.columns):  # fallback parse thủ công
    lines = open(CAP, encoding="utf-8").read().splitlines()[1:]
    recs = []
    for l in lines:
        if "," in l:
            img, cap = l.split(",", 1)
            recs.append((img.strip(), cap.strip()))
    rows = pd.DataFrame(recs, columns=["image", "caption"])

img_cap = rows.dropna().groupby("image").first().reset_index()  # 1 caption/ảnh
img_cap = img_cap.sample(frac=1.0, random_state=CFG["SEED"]).reset_index(drop=True)

# Tìm thư mục ảnh: mount Kaggle có thể lồng sâu, không nằm cạnh captions.txt
IMG_DIR = None
for name in img_cap["image"].head(10):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if hits:
        IMG_DIR = os.path.dirname(hits[0])
        break
assert IMG_DIR, "KHÔNG tìm thấy file ảnh nào dưới /kaggle/input — dataset thiếu ảnh?"
print("images dir:", IMG_DIR)

img_cap["path"] = img_cap["image"].apply(lambda s: os.path.join(IMG_DIR, s))
img_cap = img_cap[img_cap["path"].apply(os.path.exists)].reset_index(drop=True)
assert len(img_cap) > 100, f"Chỉ còn {len(img_cap)} ảnh tồn tại — sai thư mục ảnh?"
print("Ảnh có sẵn:", len(img_cap), "| caption mẫu:", img_cap["caption"].iloc[0][:80])

train_df = img_cap.iloc[:CFG["MAX_TRAIN_IMAGES"]].reset_index(drop=True)
gallery_df = img_cap.iloc[CFG["MAX_TRAIN_IMAGES"]:CFG["MAX_TRAIN_IMAGES"] + CFG["N_GALLERY"]].reset_index(drop=True)
print("train:", len(train_df), "| gallery (held-out):", len(gallery_df))"""),

code("""import torch.nn as nn
from PIL import Image

# Mapping network: image embedding (joint space) -> pseudo-word token embedding
torch.manual_seed(CFG["SEED"])
mapper = nn.Sequential(
    nn.Linear(D, D), nn.LayerNorm(D), nn.GELU(), nn.Linear(D, D),
).to(DEVICE)
print("mapper params:", sum(p.numel() for p in mapper.parameters()))

@torch.no_grad()
def encode_images(paths, bs=64):
    feats = []
    for i in range(0, len(paths), bs):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
        inputs = proc(images=imgs, return_tensors="pt").to(DEVICE)
        out = clip.vision_model(pixel_values=inputs["pixel_values"])
        feats.append(clip.visual_projection(out.pooler_output))  # [B, 512] — ổn định giữa các bản transformers
    f = torch.cat(feats)
    return f / f.norm(dim=-1, keepdim=True)

train_img_feat = encode_images(train_df["path"].tolist()).to(DEVICE)
gallery_feat = encode_images(gallery_df["path"].tolist()).to(DEVICE)
print("image features:", tuple(train_img_feat.shape), tuple(gallery_feat.shape))"""),

md("""## 2. Train mapper — InfoNCE + pseudo-label smoothing

Text input: `[BOS, pseudo-token, w2, w3, ...]` — pseudo-token ở **vị trí 1** thay embedding của từ đầu tiên, do `mapper(image_embedding)` sinh ra. CLIP text encoder bị đóng băng; gradient chỉ chảy về mapper qua embedding được chèn vào."""),

code("""import torch.nn.functional as F

# Chèn pseudo-token bằng forward hook trên token_embedding — đường chuẩn, chạy mọi bản transformers
# (không dùng inputs_embeds vì bản transformers mới xung đột input_ids/inputs_embeds)
_pseudo = {}
def _swap_pseudo(module, inputs, output):
    v = _pseudo.get("v")
    if v is None:
        return output
    out = output.clone()
    out[:, 1, :] = v          # vị trí 1 = pseudo-word
    return out

clip.text_model.embeddings.token_embedding.register_forward_hook(_swap_pseudo)

def text_forward(img_emb, ids, att, use_mapper=True):
    \"\"\"Encode câu có pseudo-token ở vị trí 1 (mapper(img_emb) nếu use_mapper, không thì raw img_emb).\"\"\"
    _pseudo["v"] = mapper(img_emb) if use_mapper else img_emb
    out = clip.text_model(input_ids=ids, attention_mask=att)
    hs = out.last_hidden_state
    eot = (ids == tok.eos_token_id).float().argmax(dim=1)   # vị trí token EOS để pooling (chuẩn CLIP)
    pooled = hs[torch.arange(hs.shape[0], device=hs.device), eot]
    tfeat = clip.text_projection(pooled)
    _pseudo.pop("v", None)
    return tfeat / tfeat.norm(dim=-1, keepdim=True)

enc_all = tok(train_df["caption"].tolist(), padding="max_length",
              truncation=True, max_length=77, return_tensors="pt")
opt = torch.optim.AdamW(mapper.parameters(), lr=CFG["LR"])
logit_scale = clip.logit_scale.exp().detach()

n = len(train_df)
for ep in range(CFG["EPOCHS"]):
    perm = torch.randperm(n)
    tot, nb = 0.0, 0
    for i in range(0, n, CFG["BATCH"]):
        idx = perm[i:i + CFG["BATCH"]]
        ie = train_img_feat[idx]
        ids = enc_all["input_ids"][idx].to(DEVICE)
        att = enc_all["attention_mask"][idx].to(DEVICE)
        if random.random() < CFG["SMOOTH_P"]:  # smoothing: caption của ảnh KHÁC trong batch
            ids = ids.roll(1, dims=0)
            att = att.roll(1, dims=0)
        tfeat = text_forward(ie, ids, att)
        logits = logit_scale * ie @ tfeat.t()
        labels = torch.arange(len(idx), device=DEVICE)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); nb += 1
    print(f"epoch {ep + 1}/{CFG['EPOCHS']}  loss={tot / nb:.4f}")
print("TRAIN DONE")"""),

md("""## 3. Eval — self-retrieval trên 200 ảnh held-out

Đo Recall@1/5: dùng chính ảnh làm query (ảnh → pseudo-token → câu template), tìm lại đúng ảnh đó trong gallery. So sánh mapper đã train vs baseline nhét **raw** image embedding vào vị trí pseudo-token (2 không gian khác nhau nên tệ — cho thấy vì sao cần mapper)."""),

code("""@torch.no_grad()
def eval_retrieval(use_mapper=True):
    B = gallery_feat.shape[0]
    enc = tok([CFG["TEMPLATE"]] * B, padding="max_length", truncation=True,
              max_length=77, return_tensors="pt")
    tfeat = text_forward(gallery_feat, enc["input_ids"].to(DEVICE),
                         enc["attention_mask"].to(DEVICE), use_mapper=use_mapper)
    ranks = (tfeat @ gallery_feat.t()).argsort(dim=1, descending=True)
    gt = torch.arange(B, device=DEVICE)
    r1 = (ranks[:, 0] == gt).float().mean().item()
    r5 = (ranks[:, :5] == gt[:, None]).any(1).float().mean().item()
    return r1, r5

import time
t0 = time.time()
R_mapper = eval_retrieval(use_mapper=True)
R_raw = eval_retrieval(use_mapper=False)
print(f"mapper (trained)  : R@1={R_mapper[0]:.3f}  R@5={R_mapper[1]:.3f}")
print(f"raw embed (no map): R@1={R_raw[0]:.3f}  R@5={R_raw[1]:.3f}")
print(f"eval xong trong {time.time() - t0:.1f}s")"""),

md("""## 4. Demo truy xuất + xuất output

Lưu ý thẳng: bản mini chỉ train để pseudo-prompt khớp **ảnh gốc**, nên template đúng nghĩa ZS-CIR ("a sketch of *", sửa thuộc tính) sẽ trôi — bài gốc train cả CC3M (~3M caption) mới tổng quát được. Đây là demo cơ chế, không phải kết quả SOTA."""),

code("""import matplotlib.pyplot as plt

@torch.no_grad()
def query_one(qi, template):
    q = gallery_feat[qi:qi + 1]
    enc = tok([template], padding="max_length", truncation=True, max_length=77, return_tensors="pt")
    tfeat = text_forward(q, enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
    sims = (tfeat @ gallery_feat.t())[0]
    top = sims.argsort(descending=True)[:4]
    return top.tolist(), sims[top].tolist()

def demo_grid(qi, template, fname):
    top, ss = query_one(qi, template)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
    axes[0].imshow(Image.open(gallery_df["path"].iloc[qi]).convert("RGB"))
    axes[0].set_title("QUERY", fontsize=10); axes[0].axis("off")
    for r, (j, s) in enumerate(zip(top, ss), start=1):
        axes[r].imshow(Image.open(gallery_df["path"].iloc[j]).convert("RGB"))
        axes[r].set_title(f"#{r} sim={s:.3f}", fontsize=9); axes[r].axis("off")
    fig.suptitle(template.replace(".", "<pseudo>"), fontsize=11)
    fig.savefig(fname, bbox_inches="tight", dpi=90)
    plt.close(fig)
    print("saved", fname)

random.seed(CFG["SEED"])
for k, qi in enumerate(random.sample(range(len(gallery_df)), 3)):
    demo_grid(qi, CFG["TEMPLATE"], f"demo_q{k}_photo.png")
print("DEMO DONE")"""),

code("""import json, shutil, pathlib

W = pathlib.Path("/kaggle/working")
metrics = {
    "model": CFG["MODEL"],
    "train_images": int(len(train_df)),
    "gallery_images": int(len(gallery_df)),
    "epochs": CFG["EPOCHS"],
    "recall_at1_mapper": R_mapper[0],
    "recall_at5_mapper": R_mapper[1],
    "recall_at1_raw_baseline": R_raw[0],
    "recall_at5_raw_baseline": R_raw[1],
}
(W / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
torch.save(mapper.state_dict(), W / "mapping.pt")
summary = "\\n".join(
    [f"PIC2WORD MINI — {metrics['train_images']} anh train, {metrics['gallery_images']} anh gallery",
     f"R@1 mapper={metrics['recall_at1_mapper']:.3f}  R@5={metrics['recall_at5_mapper']:.3f}",
     f"R@1 raw   ={metrics['recall_at1_raw_baseline']:.3f}  R@5={metrics['recall_at5_raw_baseline']:.3f}"])
(W / "summary.txt").write_text(summary + "\\n")
print(summary)
print("OUTPUT READY")"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
out = pathlib.Path(__file__).parent / "pic2word_kaggle.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("OK", out, len(cells), "cells")
