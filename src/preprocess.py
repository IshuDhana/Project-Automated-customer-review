# src/preprocess.py
from __future__ import annotations
import argparse, json, re, glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ---- config: candidate columns (lowercased) ----
TEXT_CANDS   = ["reviews.text","text","review_text"]
RATING_CANDS = ["reviews.rating","rating","stars","overall"]
DATE_CANDS   = ["reviews.date","date","review_date","time"]
CAT_CANDS    = ["categories","category"]
NAME_CANDS   = ["name","product_name","title"]
BRAND_CANDS  = ["brand","manufacturer"]
PRICE_CANDS  = ["prices.amountmin","price","prices.amountmax","price.amount"]

URL_RE  = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")

def pick(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None

def clean_text_basic(s: str) -> str:
    s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    s = HTML_RE.sub(" ", s)
    s = URL_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def map_stars(x: object, pos_min: float, neg_max: float) -> Optional[str]:
    try:
        v = float(x)
    except Exception:
        return None
    if v >= pos_min: return "positive"
    if v <= neg_max: return "negative"
    return "neutral"

def load_any(inputs: Union[str, List[str]]) -> pd.DataFrame:
    if isinstance(inputs, str):
        files = glob.glob(inputs) or ([inputs] if Path(inputs).exists() else [])
    else:
        files = []
        for p in inputs:
            files.extend(glob.glob(p) or ([p] if Path(p).exists() else []))
    if not files:
        raise FileNotFoundError(f"No input files matched: {inputs}")
    dfs = [pd.read_csv(p, low_memory=False) for p in files]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.lower()
    return df

def resolve_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "text":   pick(df, TEXT_CANDS),
        "rating": pick(df, RATING_CANDS),
        "date":   pick(df, DATE_CANDS),
        "cat":    pick(df, CAT_CANDS),
        "name":   pick(df, NAME_CANDS),
        "brand":  pick(df, BRAND_CANDS),
        "price":  pick(df, PRICE_CANDS),
    }

def standardize(
    df_raw: pd.DataFrame,
    pos_min: float,
    neg_max: float,
    min_words: int
) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    df = df_raw.copy()
    df.columns = df.columns.str.lower()
    m = resolve_mapping(df)

    # ---- text
    if m["text"]:
        df["text_raw"] = df[m["text"]].astype(str)
        df["text_clean"] = df["text_raw"].map(clean_text_basic)
    else:
        df["text_raw"] = np.nan
        df["text_clean"] = np.nan

    # canonical working text (always present)
    df["text"] = df["text_clean"].where(df["text_clean"].notna() & (df["text_clean"].str.strip() != ""), df["text_raw"])

    # lengths and filtering on canonical text
    df["text_len_words"] = df["text"].fillna("").str.split().map(lambda x: len(x) if isinstance(x, list) else 0)
    df["text_len_chars"] = df["text"].fillna("").str.len()
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    if min_words > 0:
        df = df[df["text_len_words"] >= min_words]
    df = df.drop_duplicates(subset=["text"])

    # ---- stars → sentiment
    if m["rating"]:
        df["stars"] = pd.to_numeric(df[m["rating"]], errors="coerce")
        df["sentiment"] = df["stars"].map(lambda x: map_stars(x, pos_min, neg_max))
    else:
        df["stars"] = np.nan
        df["sentiment"] = np.nan

    # ---- date
    if m["date"]:
        df["review_date"] = pd.to_datetime(df[m["date"]], errors="coerce", utc=True).dt.tz_convert(None)
    else:
        df["review_date"] = pd.NaT

    # ---- category
    if m["cat"]:
        df["category_raw"] = df[m["cat"]].astype(str)
        df["category"] = df["category_raw"].str.split(";").str[0].str.strip()
    else:
        df["category_raw"] = np.nan
        df["category"] = np.nan

    # ---- product / brand / price
    df["product_name"] = df[m["name"]].astype(str) if m["name"] else np.nan
    df["brand_name"]   = df[m["brand"]].astype(str) if m["brand"] else np.nan
    df["price"] = pd.to_numeric(df[m["price"]], errors="coerce") if m["price"] else np.nan

    # ---- stable column order
    cols = [
        "product_name","brand_name","category","review_date",
        "stars","sentiment","price",
        "text","text_raw","text_clean","text_len_words","text_len_chars"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols].reset_index(drop=True), m

def safe_to_parquet_or_csv(df: pd.DataFrame, out_dir: Path, fmt: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        path = out_dir / "clean_reviews.csv"
        df.to_csv(path, index=False)
        return path

    path = out_dir / "clean_reviews.parquet"

    # sanitize dtypes; be backward-compatible with older pandas
    try:
        clean = df.convert_dtypes(dtype_backend="numpy_nullable").copy()
    except TypeError:
        clean = df.convert_dtypes().copy()

    for col in clean.columns:
        if pd.api.types.is_string_dtype(clean[col]):
            clean[col] = clean[col].astype("object")
        elif pd.api.types.is_datetime64_any_dtype(clean[col]):
            clean[col] = pd.to_datetime(clean[col], errors="coerce").dt.tz_localize(None)

    # try pyarrow, then fastparquet, else CSV
    try:
        clean.to_parquet(path, engine="pyarrow", index=False)
        return path
    except Exception:
        try:
            import fastparquet  # noqa: F401
            clean.to_parquet(path, engine="fastparquet", index=False)
            return path
        except Exception:
            csv_path = out_dir / "clean_reviews.csv"
            clean.to_csv(csv_path, index=False)
            return csv_path

def write_sample(df: pd.DataFrame, out_dir: Path, n: int = 2000) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "clean_sample.csv"
    df.sample(min(n, len(df)), random_state=42).to_csv(p, index=False)
    return p

def write_report(df: pd.DataFrame, mapping: Dict[str, Optional[str]], out_dir: Path) -> Path:
    rep = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "date_min": str(df["review_date"].min()) if "review_date" in df else None,
        "date_max": str(df["review_date"].max()) if "review_date" in df else None,
        "sentiment_mix": df["sentiment"].value_counts(normalize=True, dropna=True).round(4).to_dict(),
        "top_categories": df["category"].value_counts(dropna=True).head(10).to_dict(),
        "median_words": int(df["text_len_words"].median()) if "text_len_words" in df else None,
        "duplicate_text_pct": float(df.duplicated(subset=["text"]).mean()),
        "column_mapping": mapping,
        "text_column": "text",  # canonical
    }
    p = out_dir / "preprocess_report.json"
    p.write_text(json.dumps(rep, indent=2))
    return p

def parse_args():
    ap = argparse.ArgumentParser(description="Standardize review CSVs for ML.")
    ap.add_argument("--input", required=True, help="CSV path or glob, e.g. 'data/*.csv'")
    ap.add_argument("--out", default="artifacts", help="Output directory")
    ap.add_argument("--out-format", choices=["parquet","csv"], default="parquet", help="Preferred output format")
    ap.add_argument("--sample-rows", type=int, default=2000, help="Sample size for clean_sample.csv")
    ap.add_argument("--pos-min", type=float, default=4.0, help="Stars >= pos_min → positive")
    ap.add_argument("--neg-max", type=float, default=2.0, help="Stars <= neg_max → negative")
    ap.add_argument("--min-words", type=int, default=1, help="Drop reviews with fewer words")
    return ap.parse_args()

def main():
    args = parse_args()

    raw = load_any(args.input)
    std, mapping = standardize(raw, pos_min=args.pos_min, neg_max=args.neg_max, min_words=args.min_words)

    out_dir = Path(args.out)
    main_path = safe_to_parquet_or_csv(std, out_dir, fmt=args.out_format)
    samp_path = write_sample(std, out_dir, n=args.sample_rows)
    rep_path  = write_report(std, mapping, out_dir)

    print(f"Saved main → {main_path}")
    print(f"Saved sample → {samp_path}")
    print(f"Saved report → {rep_path}")
    print("Done.")

if __name__ == "__main__":
    main()
