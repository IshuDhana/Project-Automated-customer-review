from __future__ import annotations
import os, re, math, json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import gradio as gr
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM
)

# ---------- Config defaults ----------
DEFAULT_CLEAN     = "artifacts/clean_reviews.parquet"
DEFAULT_CLUSTERS  = "artifacts/cluster_assignments_optimized.parquet"  # v2.1: Optimized clusters
DEFAULT_LABELS    = "artifacts/pred_labels.parquet"
DEFAULT_SUM_DIR   = "artifacts/summaries"
DEFAULT_SHOTS     = "prompts/examples_v3_single.json"  # v2.2: Using 1 example to fit within 512 token limit
DEFAULT_MODEL     = "google/flan-t5-large"
DEFAULT_DEVICE    = "cpu"   # safer on mac; switch to "mps" or "cuda" if stable
DEFAULT_DTYPE     = "auto"  # auto|fp32|fp16|bf16

# ---------- Small helpers ----------
def load_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return pd.read_parquet(p) if p.suffix.lower() != ".csv" else pd.read_csv(p)

def pick_text_col(df: pd.DataFrame) -> str:
    for c in ["text_clean", "text", "text_raw"]:
        if c in df.columns:
            return c
    raise ValueError("No text column among: text_clean, text, text_raw")

def safe_name(s: str, max_len: int = 80) -> str:
    s = re.sub(r"[^\w\-]+", "_", str(s)).strip("_")
    return s[:max_len] or "cluster"

# keyword mining for pros/cons
STOP = {
    "the","a","an","and","or","to","of","in","on","for","with","this","that","is","it","was","are","as",
    "at","my","we","you","they","our","their","but","so","very","really","just","i","me","he","she",
    "be","have","has","had","do","did","does","if","not","no","yes","too","also","can","could","would",
    "will","one","two","more","most","less","least","than","then","when","while","because","about","from",
    "by","into","over","under","up","down","out","only","any","all","there","here","what","which","who",
    "dont","doesnt","didnt","cant","couldnt","wouldnt","wont","its","im","ive","youre","youve"
}
TOKEN_RE = re.compile(r"[A-Za-z']+")

def top_words(texts: List[str], k=8) -> List[str]:
    counts: Dict[str,int] = {}
    for t in texts:
        for w in TOKEN_RE.findall(str(t).lower()):
            w = w.strip("'")
            if len(w) < 3 or w in STOP:
                continue
            counts[w] = counts.get(w, 0) + 1
    return [w for w,_ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:k]]

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

    # Handle sentiment labels properly
    lbl_col = "pred_label" if "pred_label" in df.columns else ("label" if "label" in df.columns else ("sentiment" if "sentiment" in df.columns else None))

    def agg_grp(g: pd.DataFrame) -> pd.Series:
        n = int(len(g))
        mean_stars = _safe_mean(g["stars"]) if "stars" in g.columns else np.nan

        pos = neg = neutral = 0
        if lbl_col:
            if g[lbl_col].dtype == 'object':  # text labels
                pos = int((g[lbl_col] == "positive").sum())
                neg = int((g[lbl_col] == "negative").sum()) 
                neutral = int((g[lbl_col] == "neutral").sum())
            else:  # numeric labels
                pos = int((g[lbl_col] == 2).sum())
                neg = int((g[lbl_col] == 0).sum())
                neutral = int((g[lbl_col] == 1).sum())

        examples = []
        if text_col in g.columns:
            examples = g.loc[g[text_col].notna(), text_col].head(3).tolist()

        return pd.Series(
            {
                "n_reviews": n,
                "mean_stars": mean_stars if not np.isnan(mean_stars) else None,
                "avg_stars": mean_stars if not np.isnan(mean_stars) else None,
                "pos": pos,
                "neg": neg,
                "neutral": neutral,
                "examples": examples,
            }
        )

    # Group by product name, but clean it first to handle duplicates
    if "product_name" in df.columns:
        # Clean product names to reduce near-duplicates
        df['clean_product_name'] = df['product_name'].fillna('Unknown Product').str.strip()
        # Remove very long or garbled names
        df.loc[df['clean_product_name'].str.len() > 100, 'clean_product_name'] = df.loc[df['clean_product_name'].str.len() > 100, 'clean_product_name'].str[:100] + '...'
        
        key = ["clean_product_name"]
        agg = df.groupby(key, dropna=False).apply(agg_grp, include_groups=False).reset_index()
        agg = agg.rename(columns={"clean_product_name": "product_name"})
    else:
        key = [text_col]
        agg = df.groupby(key, dropna=False).apply(agg_grp, include_groups=False).reset_index()
    
    agg["pos_share"] = (agg["pos"] / agg["n_reviews"]).fillna(0.0)
    agg["neg_share"] = (agg["neg"] / agg["n_reviews"]).fillna(0.0)
    
    return agg


def select_top(agg: pd.DataFrame, top_n: int, min_reviews: int) -> pd.DataFrame:
    # Filter by minimum reviews, but be more flexible for diverse categories
    min_threshold = max(min_reviews // 3, 10)  # Reduce threshold for better diversity
    cand = agg[agg["n_reviews"] >= min_threshold].copy()
    
    if cand.empty:
        # If still empty, take top products regardless of review count
        cand = agg.copy()
    
    # Sort by a combination of positive share and review count, but cap review dominance
    cand["score"] = (cand["pos_share"] * 0.7) + (np.log1p(cand["n_reviews"]) / np.log1p(cand["n_reviews"].max()) * 0.3)
    cand = cand.sort_values(["score", "n_reviews"], ascending=[False, False])
    
    # Try to get diverse products - avoid very similar names
    selected = []
    seen_patterns = set()
    
    for _, row in cand.iterrows():
        product_name = str(row.get("product_name", "")).lower()
        # Create a simplified pattern to avoid very similar products
        pattern = ' '.join(sorted(set(product_name.split()[:3])))  # First 3 unique words
        
        if pattern not in seen_patterns or len(selected) < 3:  # Always get at least 3
            selected.append(row)
            seen_patterns.add(pattern)
            
        if len(selected) >= top_n:
            break
    
    result = pd.DataFrame(selected) if selected else cand.head(top_n)
    return result.drop(columns=['score'], errors='ignore')


def _fmt_num(v, nd=2):
    return "NA" if pd.isna(v) else f"{float(v):.{nd}f}"

def extract_customer_quotes(df_slice: pd.DataFrame, text_col: str = "text_clean", 
                           n_positive: int = 2, n_negative: int = 2) -> Dict[str, List[str]]:
    """Extract actual customer quotes from reviews for v2.0"""
    quotes = {"positive": [], "negative": []}
    
    # Get sentiment label column
    lbl_col = "pred_label" if "pred_label" in df_slice.columns else ("label" if "label" in df_slice.columns else None)
    if not lbl_col or text_col not in df_slice.columns:
        return quotes
    
    # Extract positive quotes (shorter, concise ones are better)
    pos_reviews = df_slice[df_slice[lbl_col] == "positive"][text_col] if df_slice[lbl_col].dtype == 'object' else df_slice[df_slice[lbl_col] == 2][text_col]
    if len(pos_reviews) > 0:
        # Get reviews between 50-200 chars (not too short, not too long)
        pos_candidates = pos_reviews[pos_reviews.str.len().between(50, 200)]
        if len(pos_candidates) >= n_positive:
            quotes["positive"] = pos_candidates.sample(min(n_positive, len(pos_candidates)), random_state=42).str[:150].tolist()
        elif len(pos_reviews) > 0:
            quotes["positive"] = pos_reviews.head(n_positive).str[:150].tolist()
    
    # Extract negative quotes
    neg_reviews = df_slice[df_slice[lbl_col] == "negative"][text_col] if df_slice[lbl_col].dtype == 'object' else df_slice[df_slice[lbl_col] == 0][text_col]
    if len(neg_reviews) > 0:
        neg_candidates = neg_reviews[neg_reviews.str.len().between(50, 200)]
        if len(neg_candidates) >= n_negative:
            quotes["negative"] = neg_candidates.sample(min(n_negative, len(neg_candidates)), random_state=42).str[:150].tolist()
        elif len(neg_reviews) > 0:
            quotes["negative"] = neg_reviews.head(n_negative).str[:150].tolist()
    
    return quotes

def extract_top_complaints(df_full: pd.DataFrame, product_name: str, top_n: int = 3) -> List[str]:
    """Extract top complaints for a specific product from negative reviews."""
    if df_full is None or len(df_full) == 0:
        return []
    
    # Determine which columns are available
    rating_col = 'stars' if 'stars' in df_full.columns else 'reviews.rating'
    text_col = 'text_clean' if 'text_clean' in df_full.columns else ('text' if 'text' in df_full.columns else 'reviews.text')
    
    # Filter to this product's negative reviews (1-2 stars)
    product_negatives = df_full[
        (df_full['product_name'] == product_name) & 
        (df_full[rating_col].isin([1, 2]))
    ]
    
    if len(product_negatives) == 0:
        return []
    
    # Get longest negative review texts (more detailed complaints)
    complaints = product_negatives[text_col].dropna()
    if len(complaints) == 0:
        return []
    
    # Sort by length and take top N most detailed complaints
    complaints_sorted = complaints.apply(lambda x: str(x)[:150]).sort_values(
        key=lambda x: x.str.len(), ascending=False
    ).head(top_n).tolist()
    
    return complaints_sorted


def build_facts_block(cluster_name: str, topk: pd.DataFrame, df_full: pd.DataFrame = None) -> str:
    total_products = len(topk)
    total_reviews = topk["n_reviews"].sum()
    
    lines = [f"Write a buyer's guide for '{cluster_name}'.\nFacts:"]
    
    for _, r in topk.iterrows():
        stars_val = r["avg_stars"] if "avg_stars" in r and pd.notna(r["avg_stars"]) else r.get("mean_stars", np.nan)

        avg_s = _fmt_num(stars_val, 1)

        pname = str(r.get("product_name", "Unknown"))[:80]  # Truncate long names
        nrev = int(r.get("n_reviews", 0) or 0)
        pos = int(r.get("pos", 0) or 0)
        neg = int(r.get("neg", 0) or 0)
        
        pos_pct = f"{(pos/nrev)*100:.0f}%" if nrev > 0 else "0%"
        
        lines.append(f"- {pname}: n={nrev} reviews, {pos} positive ({pos_pct}), {neg} negative, {avg_s}★")
    
    # v2.0: Add customer quotes if full dataframe provided
    if df_full is not None and len(df_full) > 0:
        quotes = extract_customer_quotes(df_full)
        if quotes["positive"] or quotes["negative"]:
            lines.append("\nCustomer Quotes:")
            if quotes["positive"]:
                lines.append(f"Positive: {' '.join(['\"' + q + '...\"' for q in quotes['positive']])}")
            if quotes["negative"]:
                lines.append(f"Negative: {' '.join(['\"' + q + '...\"' for q in quotes['negative']])}")
    
    # v2.2: Extract top complaints for each product
    if df_full is not None and len(df_full) > 0 and len(topk) > 0:
        lines.append("\nTop Customer Complaints by Product:")
        for _, r in topk.head(3).iterrows():  # Only top 3 products
            pname = str(r.get("product_name", "Unknown"))
            complaints = extract_top_complaints(df_full, pname, top_n=3)
            if complaints:
                lines.append(f"\n{pname}:")
                for complaint in complaints:
                    lines.append(f"  - \"{complaint}\"")
    
    # Add category insights
    if total_products > 1:
        avg_rating = topk["avg_stars"].mean()
        best_rated = topk.loc[topk["avg_stars"].idxmax(), "product_name"] if not topk["avg_stars"].isna().all() else "N/A"
        worst_rated = topk.loc[topk["avg_stars"].idxmin(), "product_name"] if not topk["avg_stars"].isna().all() else "N/A"
        
        # v2.2: Identify worst product with detailed stats
        if not topk["avg_stars"].isna().all():
            worst_idx = topk["avg_stars"].idxmin()
            worst_product = topk.loc[worst_idx]
            worst_name = str(worst_product["product_name"])[:50]
            worst_rating = _fmt_num(worst_product["avg_stars"], 1)
            worst_neg_pct = f"{(worst_product['neg']/worst_product['n_reviews'])*100:.0f}%" if worst_product['n_reviews'] > 0 else "0%"
            
            lines.append(f"\nCategory Overview: {total_products} products analyzed, {total_reviews:,} total reviews")
            lines.append(f"Best rated: {str(best_rated)[:50]}")
            lines.append(f"WORST PRODUCT TO AVOID: {worst_name} ({worst_rating}★, {worst_neg_pct} negative reviews)")
            
            # Add specific worst product complaints
            if df_full is not None:
                worst_complaints = extract_top_complaints(df_full, worst_product["product_name"], top_n=2)
                if worst_complaints:
                    lines.append(f"Why avoid: {' | '.join(worst_complaints)}")
        
        # Common themes based on product names
        all_words = ' '.join(topk['product_name'].fillna('').astype(str)).lower()
        common_words = []
        for word in ['fire', 'echo', 'kindle', 'tablet', 'hd', 'wifi', 'display']:
            if word in all_words:
                common_words.append(word)
        if common_words:
            lines.append(f"Common features: {', '.join(common_words[:5])}")
    
    return "\n".join(lines) + "\n"



SYSTEM_INSTR = (
    "Generate a buyer's guide with this EXACT structure:\n\n"
    "## 🏆 Our Top 3 Recommendations\n\n"
    "<div style=\"border: 3px solid #3b82f6; border-radius: 16px; padding: 2rem; margin: 1.5rem 0; background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); box-shadow: 0 4px 20px rgba(59, 130, 246, 0.15);\">\n\n"
    "### 🥇 #1 HIGHEST RATED: **[PRODUCT NAME IN ALL CAPS]**\n\n"
    "![Product Image](https://via.placeholder.com/400x300/3b82f6/ffffff?text=[Product+Name])\n\n"
    "**Rating: [X.X]★ | Reviews: [N] | [XX]% Positive**\n\n"
    "**Why This Wins:** [Explain specific features customers love - mention actual specs like screen size, battery life, storage. Reference real customer feedback.]\n\n"
    "</div>\n\n"
    "<div style=\"border: 3px solid #8b5cf6; border-radius: 16px; padding: 2rem; margin: 1.5rem 0; background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%); box-shadow: 0 4px 20px rgba(139, 92, 246, 0.15);\">\n\n"
    "### 🥈 #2 BEST REVIEWED: **[PRODUCT NAME IN ALL CAPS]**\n\n"
    "![Product Image](https://via.placeholder.com/400x300/8b5cf6/ffffff?text=[Product+Name])\n\n"
    "**Rating: [X.X]★ | Reviews: [N] | [XX]% Positive**\n\n"
    "**Why This Wins:** [Explain why customers consistently praise this - cite specific features and benefits mentioned in reviews.]\n\n"
    "</div>\n\n"
    "<div style=\"border: 3px solid #ec4899; border-radius: 16px; padding: 2rem; margin: 1.5rem 0; background: linear-gradient(135deg, #fdf2f8 0%, #fce7f3 100%); box-shadow: 0 4px 20px rgba(236, 72, 153, 0.15);\">\n\n"
    "### 🥉 #3 MOST POPULAR: **[PRODUCT NAME IN ALL CAPS]**\n\n"
    "![Product Image](https://via.placeholder.com/400x300/ec4899/ffffff?text=[Product+Name])\n\n"
    "**Rating: [X.X]★ | Reviews: [N] | [XX]% Positive**\n\n"
    "**Why This Wins:** [Explain why this is popular - mention what customers specifically appreciate about it.]\n\n"
    "</div>\n\n"
    "## 🔍 When to Choose Each\n"
    "• Choose [Product A] if: [Specific use case with product details - e.g., 'you want 10-hour battery life and 32GB storage for heavy reading']\n"
    "• Choose [Product B] if: [Specific use case - e.g., 'you prioritize the 7-inch display and wireless charging']\n"
    "• Choose [Product C] if: [Specific use case - e.g., 'you need the most affordable option with basic features']\n\n"
    "## ⚠️ What Customers Actually Complain About\n\n"
    "**[Product A]:** [Top 2-3 specific complaints from reviews - e.g., 'Battery drains in 6 hours with WiFi on', 'Screen too dim outdoors', '16GB fills up quickly']\n"
    "**[Product B]:** [Top 2-3 complaints]\n"
    "**[Product C]:** [Top 2-3 complaints]\n\n"
    "## ❌ Product to Avoid\n\n"
    "**[Worst Product Name]: [Brief reason]** - [Explain specific issues from customer reviews - poor rating, high negative %, common failures. Recommend better alternative.]\n\n"
    "## The Bottom Line\n"
    "[Summarize why the #1 product wins with specific stats - e.g., 'With 4.8 stars and 95% positive reviews, [Product] dominates because of its [specific feature] and [specific benefit].'  Mention concrete details, not generic praise.]"
)

def load_nshots(path: str | Path) -> List[Dict[str,str]]:
    data = json.loads(Path(path).read_text())
    
    # Handle both formats: direct list or wrapped with "examples" key
    if isinstance(data, dict) and "examples" in data:
        shots = data["examples"]
    elif isinstance(data, list):
        shots = data
    else:
        raise ValueError("shots file must be a JSON list or dict with 'examples' key")
    
    if not isinstance(shots, list):
        raise ValueError("shots must be a list of {input, output}")
    
    for i, ex in enumerate(shots):
        if not isinstance(ex, dict) or "input" not in ex or "output" not in ex:
            raise ValueError(f"shots[{i}] must contain 'input' and 'output'")
    return shots

def build_prompt(shots: List[Dict[str,str]], facts: str) -> str:
    parts = [SYSTEM_INSTR, ""]
    for ex in shots:
        parts += ["Example", "Input:", ex["input"].strip(), "Output:", ex["output"].strip(), ""]
    parts += ["NOW FOLLOW THE EXACT FORMAT:", "Input:", facts.strip(), "Output:"]
    return "\n".join(parts)

def force_structured_format(text: str, facts: str) -> str:
    """Force any generated text into the structured buyer's guide format"""
    # Extract product names from facts
    import re
    products = []
    for line in facts.split('\n'):
        if ': n=' in line:
            name = line.split(':')[0].strip('- ')
            products.append(name)
    
    if len(products) == 0:
        return text  # Can't structure without products
    
    # If already structured, return as-is
    if '## 🏆 Our Top 3' in text and '**🥇 Best Overall:' in text:
        return text
        
    # Force structure with available products
    result = "## 🏆 Our Top 3 Recommendations\n\n"
    
    for i, product in enumerate(products[:3]):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        title = "Best Overall" if i == 0 else "Best Value" if i == 1 else "Specialty Pick"
        
        # Extract rating/price from facts
        rating = "4.5"
        reviews = "1,000" 
        price = "Price varies"
        
        for line in facts.split('\n'):
            if product in line and ': n=' in line:
                parts = line.split(',')
                for part in parts:
                    if '★' in part:
                        rating = part.split('★')[0].strip()
                    if 'n=' in part:
                        reviews = part.split('n=')[1].split()[0].strip()
                    if '$' in part:
                        price = part.strip()
        
        result += f"**{medal} {title}: {product}**\n"
        result += f"Rating: {rating}★ | Reviews: {reviews} | Price: {price}\n"
        result += f"Why: Excellent choice for most users.\n\n"
    
    result += "## 🔍 When to Choose Each\n"
    for product in products[:3]:
        result += f"• Choose {product} if: you need reliable performance\n"
    
    result += "\n## The Bottom Line\n"
    result += f"{products[0]} offers the best overall experience for most users."
    
    return result

def is_seq2seq(model_name: str) -> bool:
    name = model_name.lower()
    return any(k in name for k in ["t5", "flan", "ul2", "mt5"])

# lazy singletons
_LLM_CACHE = {}

def load_llm(model_name: str, device: str, dtype: str):
    key = (model_name, device, dtype)
    if key in _LLM_CACHE:
        return _LLM_CACHE[key]
    dev = torch.device(device if device in ["cuda", "cpu", "mps"] else "cpu")
    dtype_map = {"auto": None, "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[dtype]

    tok = AutoTokenizer.from_pretrained(model_name)
    if is_seq2seq(model_name):
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    else:
        mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    mdl.to(dev)
    _LLM_CACHE[key] = (tok, mdl, dev)
    return tok, mdl, dev

def get_category_temperature(category_name: str) -> float:
    """
    Adjust temperature based on category type for v2.0:
    - Technical/Specs-driven categories: Lower temperature (0.5-0.6) for precision
    - Lifestyle/Experience categories: Higher temperature (0.7-0.8) for creativity
    """
    category_lower = category_name.lower()
    
    # Technical categories - need precise, factual language
    technical_keywords = ['fire tv', 'streaming', 'battery', 'batteries', 'power', 'cable', 
                          'hdmi', 'usb', 'computer', 'accessories', 'technical']
    if any(kw in category_lower for kw in technical_keywords):
        return 0.55
    
    # Lifestyle/Experience categories - can be more creative
    lifestyle_keywords = ['echo', 'speaker', 'alexa', 'smart home', 'kindle', 'e-reader', 
                          'reading', 'entertainment']
    if any(kw in category_lower for kw in lifestyle_keywords):
        return 0.75
    
    # Tablets/screens - middle ground
    tablet_keywords = ['tablet', 'fire hd', 'fire 7', 'screen']
    if any(kw in category_lower for kw in tablet_keywords):
        return 0.65
    
    # Default: balanced creativity
    return 0.7

@torch.no_grad()
def generate_text(tok, mdl, dev, prompt: str, max_new_tokens=600,
                  temperature=0.1, top_p=0.9, do_sample=False, category_name: str = "") -> str:
    # DEBUG: Log prompt stats
    print(f"\n{'='*80}")
    print(f"DEBUG: Prompt length: {len(prompt)} chars")
    print(f"DEBUG: Max new tokens: {max_new_tokens}")
    print(f"DEBUG: Temperature: {temperature}")
    print(f"DEBUG: First 500 chars:\n{prompt[:500]}")
    print(f"DEBUG: Last 500 chars:\n{prompt[-500:]}")
    print(f"{'='*80}\n")
    
    # Don't truncate the prompt - let the model handle it
    enc = tok(prompt, return_tensors="pt", truncation=False)
    enc = {k: v.to(dev) for k, v in enc.items()}
    
    actual_tokens = enc['input_ids'].shape[1]
    print(f"DEBUG: Actual input tokens: {actual_tokens}")
    
    # flan-t5-large max is 512 tokens - must truncate if over
    # Leave ~200 tokens for generation (512 - 312 = 200)
    if actual_tokens > 312:
        print(f"WARNING: Prompt too long ({actual_tokens} tokens), truncating to 312 tokens...")
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=312)
        enc = {k: v.to(dev) for k, v in enc.items()}
    
    # Apply category-specific temperature if provided
    if category_name and do_sample:
        temperature = get_category_temperature(category_name)
    
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.5,  # Strong anti-repetition
        no_repeat_ngram_size=4,  # No repeating 4-grams
        early_stopping=True,     # Stop when done
        num_beams=2,            # Use beam search for structure
        length_penalty=0.8      # Encourage conciseness
    )
    
    if do_sample:
        gen_kwargs.update(dict(
            do_sample=True, 
            temperature=max(temperature, 0.3),  # Minimum temperature 
            top_p=top_p,
            top_k=50  # Limit vocabulary 
        ))
    else:
        gen_kwargs.update(dict(do_sample=False))
    
    if is_seq2seq(tok.name_or_path):
        out = mdl.generate(**enc, **gen_kwargs)
    else:
        out = mdl.generate(**enc, pad_token_id=tok.eos_token_id, **gen_kwargs)
    
    text = tok.decode(out[0], skip_special_tokens=True)
    print(f"DEBUG: Generated text length: {len(text)} chars")
    print(f"DEBUG: First 500 chars of output:\n{text[:500]}")
    print(f"DEBUG: Last 500 chars of output:\n{text[-500:]}")
    
    split = text.rsplit("Output:", 1)
    final = split[-1].strip() if len(split) > 1 else text.strip()
    print(f"DEBUG: Final output length: {len(final)} chars")
    print(f"{'='*80}\n")
    
    return final

# ---------- Core app actions ----------
def list_clusters(clean_path, clusters_path, labels_path, cluster_col="meta_cluster"):
    try:
        df = load_table(clean_path)
        cl = load_table(clusters_path)
        lb = load_table(labels_path)

        if cluster_col not in cl.columns:
            raise gr.Error(f"'{cluster_col}' not in {clusters_path}")
        if "pred_label" not in lb.columns and "label" not in lb.columns:
            raise gr.Error(f"Neither 'pred_label' nor 'label' found in {labels_path}")

        if len(df) != len(cl) or len(df) != len(lb):
            raise gr.Error("Row count mismatch among clean/clusters/labels. Ensure same origin and order.")

        clusters = list(pd.Series(cl[cluster_col]).dropna().astype(str).unique())
        clusters.sort()
        
        total_reviews = len(df)
        success_msg = f"""<div style="background: #d4edda; color: #155724; padding: 1rem; border-radius: 8px; border: 1px solid #c3e6cb;">
<strong>✅ Data Loaded Successfully!</strong><br>
📊 Found <strong>{len(clusters)}</strong> product categories<br>
📝 Analyzed <strong>{total_reviews:,}</strong> customer reviews<br>
🎯 Ready to generate buyer's guides!
</div>"""
        
        return gr.update(choices=clusters, value=(clusters[0] if clusters else None)), success_msg
        
    except Exception as e:
        error_msg = f"""<div style="background: #f8d7da; color: #721c24; padding: 1rem; border-radius: 8px; border: 1px solid #f5c6cb;">
<strong>❌ Error Loading Data</strong><br>
{str(e)}<br>
💡 Check your file paths in the Advanced Configuration tab.
</div>"""
        return gr.update(choices=[], value=None), error_msg

def load_existing_summary(sum_dir, cluster_name):
    if not cluster_name:
        return "Please select a product category from the dropdown above.", """<div style="background: #fff3cd; color: #856404; padding: 1rem; border-radius: 8px; border: 1px solid #ffeaa7;">
<strong>ℹ️ No Category Selected</strong><br>
Please choose a product category to view its saved review.
</div>"""
    
    path = Path(sum_dir) / f"{safe_name(cluster_name)}.md"
    if path.exists():
        content = path.read_text()
        success_msg = f"""<div style="background: #d4edda; color: #155724; padding: 1rem; border-radius: 8px; border: 1px solid #c3e6cb;">
<strong>✅ Review Loaded!</strong><br>
📁 File: <code>{path.name}</code><br>
📂 Category: <strong>{cluster_name}</strong>
</div>"""
        return content, success_msg
    
    not_found_msg = f"""<div style="background: #f8d7da; color: #721c24; padding: 1rem; border-radius: 8px; border: 1px solid #f5c6cb;">
<strong>❌ No Saved Review Found</strong><br>
No saved review exists for <strong>{cluster_name}</strong>.<br>
💡 Try generating a new review in the "Generate New Review" tab.
</div>"""
    return "", not_found_msg

def generate_cluster_summary(clean_path, clusters_path, labels_path, sum_dir, shots_path,
                             model_id, device, dtype, cluster_name,
                             min_reviews, top_n, max_new_tokens, temperature, top_p, do_sample):
    if not cluster_name:
        raise gr.Error("Select a cluster.")

    # Progress tracking
    progress_msg = "🔄 Loading data files..."
    yield "", progress_msg, ""
    
    df = load_table(clean_path)
    cl = load_table(clusters_path)
    lb = load_table(labels_path)

    progress_msg = "🔍 Processing reviews..."
    yield "", progress_msg, ""
    
    text_col = pick_text_col(df)
    df = df.copy()
    
    # Handle both pred_label (text) and label (numeric) columns
    if "pred_label" in lb.columns:
        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        df["label"] = lb["pred_label"].map(label_map).astype(int).values
    else:
        df["label"] = lb["label"].astype(int).values
    
    df["meta_cluster_name"] = cl["meta_cluster_name"].values

    g = df[df["meta_cluster_name"].astype(str) == str(cluster_name)]
    if g.empty:
        yield "", "", f"No rows for cluster {cluster_name}"
        return

    agg = aggregate_products(g, text_col=text_col)
    topk = select_top(agg, top_n=int(top_n), min_reviews=int(min_reviews))

    progress_msg = "📝 Building prompt with examples..."
    yield "", progress_msg, ""
    
    # v2.0: Pass full dataframe for quote extraction
    facts = build_facts_block(cluster_name, topk, df_full=g)
    shots = load_nshots(shots_path)
    prompt = build_prompt(shots, facts)

    progress_msg = f"🤖 Loading model ({model_id})..."
    yield "", progress_msg, ""
    
    tok, mdl, dev = load_llm(model_id, device, dtype)
    
    progress_msg = "✨ Generating comprehensive buyer's guide..."
    yield "", progress_msg, ""
    
    md_body = generate_text(tok, mdl, dev, prompt,
                            max_new_tokens=int(max_new_tokens),
                            temperature=float(temperature),
                            top_p=float(top_p),
                            do_sample=bool(do_sample),
                            category_name=cluster_name)
    
    # v2.2: DO NOT force format - model generates proper HTML divs with images
    # md_body = force_structured_format(md_body, facts)

    # Enhanced header with stats and styling - LIGHT THEME
    rows = len(g)
    pos_share = (g["label"] == 2).mean() if rows else 0.0
    neg_share = (g["label"] == 0).mean() if rows else 0.0
    avg_stars = g["stars"].mean() if "stars" in g.columns and len(g) > 0 else 0.0
    
    # Create a light-themed header
    header = f"""<div style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 50%, #ffffff 100%); 
                      color: #1a1a1a; 
                      padding: 3rem 2rem; 
                      border-radius: 20px; 
                      margin-bottom: 2rem; 
                      border: 2px solid #e0e0e0;
                      box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
    <div style="position: relative; z-index: 2;">
        <h1 style="color: #1a1a1a;
                   margin: 0; 
                   font-size: 2.5rem; 
                   font-weight: 800;
                   text-align: center;
                   letter-spacing: -1px;">
            {cluster_name}
        </h1>
        <p style="font-size: 1rem; 
                  margin: 0.5rem 0; 
                  text-align: center;
                  text-transform: uppercase;
                  letter-spacing: 2px;
                  color: #666;">
            AI-Generated Buyer's Guide
        </p>
        <div style="display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); 
                    gap: 1.5rem; 
                    margin-top: 2rem;">
            <div style="text-align: center; 
                        background: #e3f2fd; 
                        padding: 1.5rem; 
                        border-radius: 12px;
                        border: 2px solid #2196f3;">
                <div style="font-size: 2.5rem; font-weight: 700; color: #1976d2;">{rows:,}</div>
                <div style="color: #666; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">Reviews</div>
            </div>
            <div style="text-align: center; 
                        background: #f3e5f5; 
                        padding: 1.5rem; 
                        border-radius: 12px;
                        border: 2px solid #9c27b0;">
                <div style="font-size: 2.5rem; font-weight: 700; color: #7b1fa2;">{pos_share:.0%}</div>
                <div style="color: #666; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">Positive</div>
            </div>
            <div style="text-align: center; 
                        background: #fff3e0; 
                        padding: 1.5rem; 
                        border-radius: 12px;
                        border: 2px solid #ff9800;">
                <div style="font-size: 2.5rem; font-weight: 700; color: #f57c00;">★{avg_stars:.1f}</div>
                <div style="color: #666; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">Rating</div>
            </div>
        </div>
    </div>
</div>

"""
    
    # Use the generated content directly without extra styling
    # The model output already includes proper HTML divs with colored borders and images
    styled_body = f"""<div style="background: #ffffff; 
                             padding: 2rem; 
                             font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                             font-size: 18px; 
                             line-height: 1.8;
                             color: #000000;">
{md_body.strip()}
</div>

<div style="background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); 
            padding: 1.5rem; 
            border-radius: 12px; 
            margin-top: 2rem; 
            border: 2px solid #e0e0e0;">
    <p style="margin: 0; 
              color: #1976d2; 
              font-size: 14px; 
              font-weight: 600;
              display: flex;
              align-items: center;
              gap: 0.5rem;">
        <span style="font-size: 1.5rem;">🤖</span>
        <strong>AI Analysis:</strong> 
        <span style="color: #333;">
            Based on {rows:,} real customer reviews • {pos_share:.0%} positive sentiment • 
            Powered by advanced language models for informed decision-making
        </span>
    </p>
</div>"""

    final_md = header + styled_body + "\n"

    progress_msg = "💾 Saving to file..."
    yield final_md, progress_msg, ""
    
    Path(sum_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(sum_dir) / f"{safe_name(cluster_name)}.md"
    out_path.write_text(final_md)
    
    success_msg = f"""<div style="background: #d4edda; color: #155724; padding: 1rem; border-radius: 8px; border: 1px solid #c3e6cb; margin: 1rem 0;">
<strong>✅ Success!</strong> Generated comprehensive review for <strong>{cluster_name}</strong><br>
📁 Saved as: <code>{out_path.name}</code><br>
📊 Based on {rows:,} customer reviews
</div>"""
    
    yield final_md, "✅ Complete!", success_msg

# ---------- UI ----------
# Futuristic minimalist CSS
custom_css = """
/* Global theme */
.gradio-container {
    background: linear-gradient(135deg, #0c0c0c 0%, #1a1a1a 100%);
    color: #e0e0e0;
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Main interface styling */
.main-header {
    background: linear-gradient(135deg, #00d4ff 0%, #8b5cf6 50%, #f59e0b 100%);
    padding: 3rem 2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
}

/* Glass morphism panels */
.glass-panel {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem 0;
}

/* Buttons with futuristic glow */
.primary-btn {
    background: linear-gradient(45deg, #00d4ff, #8b5cf6);
    border: none;
    border-radius: 12px;
    padding: 12px 24px;
    color: white;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 8px 32px rgba(0,212,255,0.3);
}

.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0,212,255,0.4);
}

/* Output styling */
.review-output {
    background: rgba(0,0,0,0.4);
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Inputs and dropdowns */
.custom-input {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 12px;
    color: #e0e0e0;
    padding: 12px 16px;
}

.custom-input:focus {
    border-color: #00d4ff;
    box-shadow: 0 0 20px rgba(0,212,255,0.3);
}

/* Minimalist scrollbars */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255,255,255,0.05);
}

::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.2);
    border-radius: 4px;
}

/* Neon accents */
.neon-border {
    border: 1px solid;
    border-image: linear-gradient(45deg, #00d4ff, #8b5cf6, #f59e0b) 1;
    border-radius: 12px;
}

.generate-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem 2rem;
    border: none;
    border-radius: 25px;
    font-size: 1.1rem;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
}

.generate-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

.cluster-selector {
    background: white;
    border: 2px solid #e9ecef;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}

.status-message {
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.success {
    background: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}

.info {
    background: #d1ecf1;
    color: #0c5460;
    border: 1px solid #bee5eb;
}
"""

with gr.Blocks(title="🤖 RoboReviews - AI Product Review Generator", css=custom_css, theme=gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="purple", 
    neutral_hue="slate",
    text_size="md"
).set(
    body_background_fill="white",
    body_text_color="#1f2937",
    block_background_fill="#f9fafb",
    block_title_text_color="#111827"
)) as demo:
    
    # Main Header
    gr.HTML("""
    <div style="background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%); 
                color: white; 
                padding: 3rem 2rem; 
                border-radius: 20px; 
                margin-bottom: 2rem; 
                text-align: center;
                box-shadow: 0 10px 40px rgba(59, 130, 246, 0.25);
                position: relative;
                overflow: hidden;">
        <div style="position: absolute; 
                    top: -50%; 
                    left: -50%; 
                    width: 200%; 
                    height: 200%; 
                    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
                    animation: pulse 6s ease-in-out infinite;"></div>
        <div style="position: relative; z-index: 2;">
            <h1 style="margin: 0; 
                       font-size: 3.5rem; 
                       font-weight: 900;
                       color: white;
                       text-shadow: 0 2px 15px rgba(0,0,0,0.2);
                       letter-spacing: -2px;">
                🤖 RoboReviews
            </h1>
            <p style="font-size: 1.2rem; 
                      margin: 1rem 0 0.5rem 0; 
                      color: white;
                      font-weight: 600;
                      text-shadow: 0 1px 10px rgba(0,0,0,0.15);">
                AI-Powered Product Analysis
            </p>
            <p style="font-size: 1rem; 
                      margin: 0; 
                      color: rgba(255,255,255,0.95);
                      max-width: 600px;
                      margin: 0 auto;
                      line-height: 1.6;">
                Transform thousands of customer reviews into professional buyer's guides • 
                Style of The Wirecutter, Consumer Reports, The Verge
            </p>
        </div>
    </div>
    
    <style>
    @keyframes pulse {
        0%, 100% { opacity: 0.3; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.1); }
    }
    </style>
    """)

    # Quick Start Section
    with gr.Group():
        gr.Markdown("### 🚀 Quick Start")
        with gr.Row():
            with gr.Column(scale=3):
                cluster_dd = gr.Dropdown(
                    choices=[], 
                    label="📂 Select Product Category", 
                    elem_classes=["cluster-selector"]
                )
            with gr.Column(scale=1):
                load_btn = gr.Button("🔄 Load Categories", variant="secondary")
        
        load_msg = gr.Markdown("")

    # Main Content Tabs
    with gr.Tabs():
        # View Existing Reviews Tab
        with gr.TabItem("📖 View Saved Reviews", id="view_tab"):
            gr.Markdown("### Browse Previously Generated Reviews")
            view_btn = gr.Button("📄 Open Saved Review", variant="primary", size="lg")
            md_view = gr.Markdown(elem_classes=["review-output"])
            view_status = gr.Markdown("")

        # Generate New Reviews Tab  
        with gr.TabItem("✨ Generate New Review", id="generate_tab"):
            gr.Markdown("### Generate Fresh Product Review")
            
            with gr.Accordion("⚙️ Generation Settings", open=False):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**📊 Content Settings**")
                        min_reviews = gr.Number(
                            value=50, 
                            label="Minimum reviews per product",
                            info="Products need at least this many reviews to be included",
                            precision=0
                        )
                        top_n = gr.Slider(
                            1, 5, 
                            value=3, 
                            step=1, 
                            label="Number of top products to feature",
                            info="How many products to highlight in the review"
                        )
                    
                    with gr.Column():
                        gr.Markdown("**🎯 AI Model Settings**")
                        max_tokens = gr.Slider(
                            512, 3072, 
                            value=1800, 
                            step=64, 
                            label="Review length (tokens)",
                            info="Longer = more detailed reviews (v2.2: needs 1500+ for full format)"
                        )
                        temperature = gr.Slider(
                            0.0, 1.5, 
                            value=0.7, 
                            step=0.05, 
                            label="Creativity level",
                            info="v2.2: Use 0.7-0.8 for proper format following"
                        )
                        top_p = gr.Slider(
                            0.1, 1.0, 
                            value=0.9, 
                            step=0.05, 
                            label="Diversity (top-p)",
                            info="Controls word choice diversity"
                        )
                        do_sample = gr.Checkbox(
                            value=True, 
                            label="Enable creative sampling",
                            info="Allows for more natural, varied text generation"
                        )

            gen_btn = gr.Button(
                "🚀 Generate Comprehensive Review", 
                variant="primary", 
                size="lg",
                elem_classes=["generate-btn"]
            )
            
            gen_progress = gr.Textbox(
                label="Generation Progress",
                value="",
                interactive=False,
                visible=True
            )
            gen_status = gr.Markdown("")
            md_out = gr.Markdown(elem_classes=["review-output"])

        # Advanced Configuration Tab
        with gr.TabItem("🔧 Advanced Configuration", id="config_tab"):
            gr.Markdown("### Advanced System Configuration")
            gr.Markdown("*Modify these settings only if you need to use custom data sources or models*")
            
            with gr.Accordion("📁 Data Sources", open=True):
                gr.Markdown("**File Paths** - Specify your data sources")
                with gr.Row():
                    clean_in = gr.Textbox(
                        value=DEFAULT_CLEAN, 
                        label="🔍 Clean reviews data",
                        info="Preprocessed customer reviews"
                    )
                    clus_in = gr.Textbox(
                        value=DEFAULT_CLUSTERS, 
                        label="📊 Product clusters",
                        info="Product categorization data"
                    )
                with gr.Row():
                    labs_in = gr.Textbox(
                        value=DEFAULT_LABELS, 
                        label="🏷️ Sentiment labels",
                        info="Sentiment analysis results"
                    )
                    sums_in = gr.Textbox(
                        value=DEFAULT_SUM_DIR, 
                        label="💾 Output directory",
                        info="Where to save generated reviews"
                    )

            with gr.Accordion("🤖 AI Model Configuration", open=False):
                gr.Markdown("**Model Settings** - Configure the AI model")
                with gr.Row():
                    shots_in = gr.Textbox(
                        value=DEFAULT_SHOTS, 
                        label="📝 Example prompts",
                        info="v2.2: Using examples_v3.json with sentiment ranking and images"
                    )
                    model_in = gr.Textbox(
                        value="google/flan-t5-large", 
                        label="🧠 AI Model",
                        info="HuggingFace model identifier"
                    )
                with gr.Row():
                    device_in = gr.Dropdown(
                        choices=["cpu", "mps", "cuda"], 
                        value=DEFAULT_DEVICE, 
                        label="⚡ Processing Device",
                        info="CPU (safe), MPS (Apple), CUDA (NVIDIA)"
                    )
                    dtype_in = gr.Dropdown(
                        choices=["auto", "fp32", "fp16", "bf16"], 
                        value=DEFAULT_DTYPE, 
                        label="🔢 Precision",
                        info="Model precision (auto recommended)"
                    )

    # Footer
    gr.HTML("""
    <div style="text-align: center; 
                margin-top: 4rem; 
                padding: 3rem 2rem; 
                background: rgba(0,0,0,0.3); 
                border-radius: 16px;
                border: 1px solid rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);">
        <p style="color: #00d4ff; 
                  font-size: 1.2rem;
                  margin: 0 0 1rem 0;
                  font-weight: 600;
                  text-transform: uppercase;
                  letter-spacing: 2px;">
            🤖 <strong>RoboReviews</strong>
        </p>
        <p style="font-size: 1rem; 
                  color: #e0e0e0; 
                  margin: 0;
                  opacity: 0.8;
                  line-height: 1.6;">
            Analyze thousands of customer reviews in seconds • Generate buyer's guides like The Wirecutter<br>
            <span style="color: #8b5cf6;">Powered by Advanced AI • Built for Consumer Intelligence</span>
        </p>
    </div>
    """)

    # wiring
    load_btn.click(
        lambda clean_path, clusters_path, labels_path: list_clusters(clean_path, clusters_path, labels_path, "meta_cluster_name"),
        inputs=[clean_in, clus_in, labs_in],
        outputs=[cluster_dd, load_msg]
    )

    view_btn.click(
        load_existing_summary,
        inputs=[sums_in, cluster_dd],
        outputs=[md_view, view_status]
    )

    gen_btn.click(
        generate_cluster_summary,
        inputs=[clean_in, clus_in, labs_in, sums_in, shots_in,
                model_in, device_in, dtype_in, cluster_dd,
                min_reviews, top_n, max_tokens, temperature, top_p, do_sample],
        outputs=[md_out, gen_progress, gen_status]
    )

if __name__ == "__main__":
    # Lower MPS high watermark if user wants to try Apple GPU; safer path is CPU.
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
    demo.launch()
