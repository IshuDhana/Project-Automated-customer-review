# src/generate_summaries.py
# Usage example:
#   python src/generate_summaries.py \
#     --clean artifacts/clean_reviews.parquet \
#     --clusters artifacts/cluster_assignments_product_based_named.parquet \
#     --labels artifacts/pred_labels.parquet \
#     --shots_path prompts/examples.json \
#     --gen_model google/flan-t5-base \
#     --out_dir artifacts/summaries \
#     --cluster_col meta_cluster \
#     --min_reviews 50 \
#     --top_n 3 \
#     --device cpu \
#     --max_new_tokens 220 \
#     --temperature 0.0 \
#     --top_p 0.9 \
#     --limit_clusters 5

from __future__ import annotations

import argparse, json, math, os, re, sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# ------------------------- utils -------------------------

def ensure_cols(df: pd.DataFrame, must: List[str], where: str):
    missing = [c for c in must if c not in df.columns]
    if missing:
        raise ValueError(f"{where}: missing columns {missing}. Has {list(df.columns)}")

def load_parquet_or_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)

def sanitize_filename(s: str, max_len: int = 120) -> str:
    s = re.sub(r"[^\w\-.]+", "_", s)
    if len(s) > max_len:
        root, ext = (s, "")
        if "." in s:
            root, ext = s.rsplit(".", 1)
            ext = "." + ext
        s = root[: max_len - len(ext)] + ext
    return s

def safe_float(x) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")

def pick_text_col(df: pd.DataFrame) -> str:
    for c in ["text_clean", "text", "text_raw"]:
        if c in df.columns:
            return c
    raise ValueError("No text column found. Expected one of: text_clean, text, text_raw")

def attach_pred_labels(df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    # Strategy:
    # 1) If same length and 'pred_label' present → align by row order.
    # 2) Else if labels_df has a join key (product_name or text_raw) → left merge.
    # 3) Else error.
    if "pred_label" not in labels_df.columns:
        raise ValueError("labels file must contain column 'pred_label'")

    out = df.copy()
    if len(labels_df) == len(df) and "pred_label" in labels_df.columns and labels_df.index.equals(labels_df.index):
        out["pred_label"] = labels_df["pred_label"].to_numpy()
        return out

    join_keys = [k for k in ["text_raw", "product_name"] if k in df.columns and k in labels_df.columns]
    if join_keys:
        jkey = join_keys[0]
        tmp = labels_df[[jkey, "pred_label"]].drop_duplicates(subset=[jkey])
        out = out.merge(tmp, on=jkey, how="left")
        return out

    # fallback: if labels has an 'row_id' used earlier
    if "row_id" in labels_df.columns:
        if "row_id" not in out.columns:
            out = out.reset_index(drop=False).rename(columns={"index": "row_id"})
        tmp = labels_df[["row_id", "pred_label"]]
        out = out.merge(tmp, on="row_id", how="left")
        return out

    raise ValueError("Could not align labels to data. Provide same length or a join key (text_raw or product_name).")

def load_shots(shots_path: str) -> str:
    p = Path(shots_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing shots file: {shots_path}")
    data = json.loads(p.read_text())
    # Expected format:
    # {
    #   "instruction": "...",
    #   "examples": [{"input":"...", "output":"..."}, ...]
    # }
    parts = []
    if "instruction" in data and data["instruction"]:
        parts.append(data["instruction"].strip())
    exs = data.get("examples", [])
    for i, ex in enumerate(exs, 1):
        inx = ex.get("input", "").strip()
        outx = ex.get("output", "").strip()
        if inx:
            parts.append(f"\n### Example {i} — input\n{inx}")
        if outx:
            parts.append(f"\n### Example {i} — output\n{outx}")
    return "\n".join(parts).strip()

def build_generator(model_name: str, device: str = "cpu", dtype_str: str = "auto"):
    tok = AutoTokenizer.from_pretrained(model_name)
    # dtype selection
    if dtype_str == "auto":
        torch_dtype = None
    elif dtype_str == "fp32":
        torch_dtype = torch.float32
    elif dtype_str == "fp16":
        torch_dtype = torch.float16
    elif dtype_str == "bf16":
        torch_dtype = torch.bfloat16
    else:
        raise ValueError("--dtype must be one of auto|fp32|fp16|bf16")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    elif device == "mps" and torch.backends.mps.is_available():
        model = model.to("mps")
    else:
        model = model.to("cpu")

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tok,
        device=0 if device == "cuda" and torch.cuda.is_available() else -1  # cpu/mps handled by .to() above
    )
    return tok, pipe

# -------------------- aggregation and prompting --------------------

import numpy as np
import pandas as pd

def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _safe_median(s: pd.Series) -> float:
    s_num = _safe_num(s)
    med = s_num.median(skipna=True)
    return float(med) if pd.notna(med) else np.nan

def _safe_mean(s: pd.Series) -> float:
    s_num = _safe_num(s)
    m = s_num.mean(skipna=True)
    return float(m) if pd.notna(m) else np.nan

def aggregate_products(df: pd.DataFrame, text_col: str = "text_clean") -> pd.DataFrame:
    df = df.copy()
    if "stars" in df.columns:
        df["stars"] = _safe_num(df["stars"])

    lbl_col = "pred_label" if "pred_label" in df.columns else ("label" if "label" in df.columns else ("sentiment" if "sentiment" in df.columns else None))

    def agg_grp(g: pd.DataFrame) -> pd.Series:
        n = int(len(g))
        mean_stars = _safe_mean(g["stars"]) if "stars" in g.columns else np.nan

        pos = neg = neu = 0
        if lbl_col:
            pos = int((g[lbl_col] == "positive").sum())
            neg = int((g[lbl_col] == "negative").sum())
            neu = int((g[lbl_col] == "neutral").sum())

        examples = []
        if text_col in g.columns:
            examples = g.loc[g[text_col].notna(), text_col].head(3).tolist()

        # Emit both names to satisfy old/new callers
        return pd.Series(
            {
                "n_reviews": n,
                "mean_stars": mean_stars if not np.isnan(mean_stars) else None,
                "avg_stars": mean_stars if not np.isnan(mean_stars) else None,
                "pos": pos,
                "neg": neg,
                "neu": neu,
                "examples": examples,
            }
        )

    key = ["product_name"] if "product_name" in df.columns else [text_col]
    agg = df.groupby(key, dropna=False).apply(agg_grp).reset_index()
    agg["pos_share"] = (agg["pos"] / agg["n_reviews"]).fillna(0.0)
    
    # Create group_key column from the groupby key
    if "product_name" in df.columns:
        agg["group_key"] = agg["product_name"]
    else:
        agg["group_key"] = agg[text_col]
    
    return agg

# Alias for backward compatibility
agg_products = aggregate_products

def select_top(agg_df: pd.DataFrame, top_n: int, min_reviews: int) -> pd.DataFrame:
    if len(agg_df) == 0:
        return agg_df
    cand = agg_df[agg_df["n_reviews"] >= min_reviews].copy()
    if len(cand) == 0:
        cand = agg_df.copy()
    return cand.head(top_n)

def sample_snippets(df_slice: pd.DataFrame,
                    key_values: List[str],
                    per_item: int = 6,
                    text_col: str = "text_clean",
                    max_chars: int = 220,
                    seed: int = 7) -> Dict[str, List[str]]:
    rs = np.random.RandomState(seed)
    bag: Dict[str, List[str]] = {}
    # choose join key
    key_col = "product_name" if "product_name" in df_slice.columns else ("brand_name" if "brand_name" in df_slice.columns else None)

    for k in key_values:
        if key_col is None:
            sub = df_slice
        else:
            sub = df_slice[df_slice[key_col].fillna("") == k]
        if len(sub) == 0:
            bag[k] = []
            continue
        take = sub.sample(min(per_item, len(sub)), random_state=rs)
        texts = take[text_col].astype(str).tolist()
        texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
        texts = [t[:max_chars] for t in texts]
        bag[k] = texts
    return bag

def build_facts_block(cluster_label: str, topk_df: pd.DataFrame) -> str:
    lines = [f"Write a buyer's guide for '{cluster_label}'.", "Facts:"]
    for _, r in topk_df.iterrows():
        price_s = "NA" if pd.isna(r.get("median_price", float("nan"))) else f"${r['median_price']:.2f}"
        mean_s  = "NA" if pd.isna(r.get("mean_stars", float("nan"))) else f"{r['mean_stars']:.1f}★"
        total_reviews = int(r['n_reviews'])
        positive_reviews = int(r.get('pos', 0))
        negative_reviews = int(r.get('neg', 0))
        lines.append(
            f"- {r['group_key']}: n={total_reviews} reviews, {positive_reviews} positive, "
            f"{negative_reviews} negative, {mean_s}, {price_s}"
        )
    return "\n".join(lines)

def build_prompt(cluster_label: str,
                 facts_md: str,
                 shot_block: str,
                 snippets: Dict[str, List[str]]) -> str:
    # Build representative reviews section
    review_parts = []
    if snippets:
        # Group positive and negative snippets
        all_positives = []
        all_negatives = []
        
        for k, lst in snippets.items():
            if not lst:
                continue
            # Assume first half are positive, second half negative (this is a simplification)
            # In practice, you might want to use sentiment to properly classify
            mid_point = len(lst) // 2
            all_positives.extend(lst[:mid_point] if mid_point > 0 else lst[:1])
            all_negatives.extend(lst[mid_point:] if mid_point > 0 else [])
        
        if all_positives:
            review_parts.append("Top Positives: " + ", ".join(all_positives[:3]))
        if all_negatives:
            review_parts.append("Top Complaints: " + ", ".join(all_negatives[:3]))
    
    # Build the complete prompt
    parts = [shot_block, "\n\n", facts_md]
    if review_parts:
        parts.extend(["\n"] + review_parts)
    
    return "\n".join(parts)

def generate_summary(prompt: str,
                     pipe,
                     tokenizer,
                     max_new_tokens: int,
                     temperature: float,
                     top_p: float,
                     do_sample: bool) -> str:
    gen = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        num_return_sequences=1,
        truncation=True,
    )
    # HF pipeline returns list of dicts
    txt = gen[0]["generated_text"] if isinstance(gen, list) else str(gen)
    return txt.strip()

# ---------------------------- cli ----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Generate per-cluster PM-style summaries with few-shot prompting.")
    ap.add_argument("--clean", default="artifacts/clean_reviews.parquet")
    ap.add_argument("--clusters", default="artifacts/cluster_assignments_product_based_final.parquet")
    ap.add_argument("--labels", default="artifacts/pred_labels.parquet")
    ap.add_argument("--cluster_col", default="meta_cluster", help="cluster or meta_cluster")
    ap.add_argument("--out_dir", default="artifacts/summaries")
    ap.add_argument("--min_reviews", type=int, default=50)
    ap.add_argument("--top_n", type=int, default=3)

    ap.add_argument("--gen_model", default="google/flan-t5-base")
    ap.add_argument("--shots_path", required=True)

    ap.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    ap.add_argument("--dtype", choices=["auto","fp32","fp16","bf16"], default="auto")

    ap.add_argument("--max_new_tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--do_sample", action="store_true")

    ap.add_argument("--limit_clusters", type=int, default=None, help="Optional cap for quick runs")
    ap.add_argument("--per_item", type=int, default=6, help="snippets per top product")
    ap.add_argument("--max_chars", type=int, default=220, help="truncate each snippet")

    return ap.parse_args()

# ---------------------------- main ----------------------------

def main():
    args = parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = load_parquet_or_csv(args.clean)
    cl = load_parquet_or_csv(args.clusters)

    # allow either 'cluster' or 'meta_cluster'
    if args.cluster_col not in cl.columns:
        # auto-map if user asked for cluster but file has meta_cluster
        if args.cluster_col == "cluster" and "meta_cluster" in cl.columns:
            cl = cl.rename(columns={"meta_cluster": "cluster"})
        else:
            raise ValueError(f"Cluster file missing '{args.cluster_col}'. Has {list(cl.columns)}")

    # Attach cluster column to df by index position or join on product_name if present
    if len(cl) == len(df):
        df[args.cluster_col] = cl[args.cluster_col].to_numpy()
    else:
        # try a safe join on product_name + text_raw if present
        join_keys = [k for k in ["product_name", "text_raw"] if k in df.columns and k in cl.columns]
        if not join_keys:
            raise ValueError("Cannot align clusters to clean data. Provide same length or a join key (product_name or text_raw).")
        jk = join_keys[0]
        df = df.merge(cl[[jk, args.cluster_col]].drop_duplicates(subset=[jk]), on=jk, how="left")

    # Attach predicted labels for sentiment if provided
    if args.labels and Path(args.labels).exists():
        labels_df = load_parquet_or_csv(args.labels)
        df = attach_pred_labels(df, labels_df)
    else:
        if "pred_label" not in df.columns:
            # fallback to existing 'sentiment' if available
            if "sentiment" in df.columns:
                df["pred_label"] = df["sentiment"]
            else:
                # neutral default when nothing is available
                df["pred_label"] = "neutral"

    # choose text column
    text_col = pick_text_col(df)

    # shots block for few-shot prompting
    shots_block = load_shots(args.shots_path)

    # build generator
    tokenizer, pipe = build_generator(args.gen_model, device=args.device, dtype_str=args.dtype)

    # cluster ids
    if args.cluster_col not in df.columns:
        raise ValueError(f"Clean data missing cluster column '{args.cluster_col}'. Columns: {list(df.columns)}")

    all_cluster_ids = list(pd.Series(df[args.cluster_col]).dropna().unique())
    # consistent order
    try:
        all_cluster_ids = sorted(all_cluster_ids)
    except Exception:
        pass

    if args.limit_clusters:
        all_cluster_ids = all_cluster_ids[: args.limit_clusters]

    for cid in all_cluster_ids:
        g = df[df[args.cluster_col] == cid].copy()
        if len(g) == 0:
            print(f"[skip] cluster {cid} empty")
            continue

        agg = agg_products(g, text_col=text_col)
        topk = select_top(agg, top_n=args.top_n, min_reviews=args.min_reviews)
        if len(topk) == 0:
            print(f"[skip] cluster {cid} no products after filter")
            continue

        facts = build_facts_block(f"{args.cluster_col}={cid}", topk)
        keys = topk["group_key"].tolist()
        snips = sample_snippets(
            g, keys, per_item=args.per_item, text_col=text_col, max_chars=args.max_chars,
            seed=(int(cid) if isinstance(cid, (int, np.integer)) else 7)
        )

        prompt = build_prompt(f"{args.cluster_col}={cid}", facts, shots_block, snips)

        out_txt = generate_summary(
            prompt=prompt,
            pipe=pipe,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
        )

        # write one file per cluster
        fname = sanitize_filename(f"{args.cluster_col}_{str(cid)}.md")
        (out_dir / fname).write_text(out_txt)
        print(f"[ok] {fname} | rows={len(g)} | top_items={len(topk)}")

    print("Done.")

if __name__ == "__main__":
    # Safety for MPS watermark issues. User can still set device=cpu explicitly.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
