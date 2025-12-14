# src/evaluate_pipeline.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, silhouette_score, adjusted_rand_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
    calinski_harabasz_score, davies_bouldin_score
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import label_binarize
from scipy import stats

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Optional libs
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None

try:
    import sacrebleu
except Exception:
    sacrebleu = None


def load_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Missing file: {p}")
    return pd.read_parquet(p) if p.suffix.lower() != ".csv" else pd.read_csv(p)


def resolve_text_col(df: pd.DataFrame) -> str:
    for c in ["text_clean", "text", "text_raw"]:
        if c in df.columns:
            return c
    raise SystemExit("No text column found among: text_clean, text, text_raw")


def weak_label_from_stars(stars: pd.Series, pos_min=4.0, neg_max=2.0) -> pd.Series:
    def m(v):
        try:
            x = float(v)
        except Exception:
            return None
        if x >= pos_min: return 2   # positive
        if x <= neg_max: return 0   # negative
        return 1                    # neutral
    return stars.map(m)


def predict_labels(texts: List[str], clf_dir: Path, device: torch.device, max_length=256, batch_size=32) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(clf_dir)
    model = AutoModelForSequenceClassification.from_pretrained(clf_dir).to(device).eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]
            enc = tok(chunk, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=int)


def eval_classifier(clean_path: Path, clf_dir: Path, device_str="cpu", sample: int=20000) -> Dict:
    df = load_table(clean_path)
    text_col = resolve_text_col(df)

    if "stars" not in df:
        raise SystemExit("Classifier eval needs 'stars' column in clean parquet.")

    # weak gold labels from stars
    y_true = weak_label_from_stars(df["stars"]).astype("float")
    mask = y_true.notna()
    df = df.loc[mask].reset_index(drop=True)
    y_true = y_true.loc[mask].astype("int").values

    if sample and len(df) > sample:
        df = df.sample(sample, random_state=42).reset_index(drop=True)
        y_true = y_true[:len(df)]

    device = torch.device("cpu") if device_str != "cuda" or not torch.cuda.is_available() else torch.device("cuda")
    y_pred = predict_labels(df[text_col].astype(str).tolist(), clf_dir, device=device)

    # Basic metrics
    acc = float(accuracy_score(y_true, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    p_c, r_c, f1_c, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    # Additional statistical metrics
    mcc = float(matthews_corrcoef(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC-AUC and PR-AUC (for multi-class)
    try:
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        y_pred_bin = label_binarize(y_pred, classes=[0, 1, 2])
        
        # Calculate AUC for each class vs rest
        auc_scores = {}
        ap_scores = {}
        classes = ["negative", "neutral", "positive"]
        
        for i, class_name in enumerate(classes):
            if len(np.unique(y_true_bin[:, i])) > 1:  # Check if both classes present
                auc_scores[class_name] = float(roc_auc_score(y_true_bin[:, i], y_pred_bin[:, i]))
                ap_scores[class_name] = float(average_precision_score(y_true_bin[:, i], y_pred_bin[:, i]))
            else:
                auc_scores[class_name] = float('nan')
                ap_scores[class_name] = float('nan')
        
        # Macro average AUC
        valid_aucs = [v for v in auc_scores.values() if not np.isnan(v)]
        macro_auc = float(np.mean(valid_aucs)) if valid_aucs else float('nan')
        
        valid_aps = [v for v in ap_scores.values() if not np.isnan(v)]
        macro_ap = float(np.mean(valid_aps)) if valid_aps else float('nan')
        
    except Exception as e:
        auc_scores = {"negative": float('nan'), "neutral": float('nan'), "positive": float('nan')}
        ap_scores = {"negative": float('nan'), "neutral": float('nan'), "positive": float('nan')}
        macro_auc = float('nan')
        macro_ap = float('nan')
    
    # Class distribution analysis
    class_counts = np.bincount(y_true, minlength=3)
    class_distribution = {
        "negative": {"count": int(class_counts[0]), "percentage": float(class_counts[0] / len(y_true) * 100)},
        "neutral": {"count": int(class_counts[1]), "percentage": float(class_counts[1] / len(y_true) * 100)},
        "positive": {"count": int(class_counts[2]), "percentage": float(class_counts[2] / len(y_true) * 100)}
    }
    
    # Error analysis - most confused pairs
    confusion_pairs = []
    for i in range(3):
        for j in range(3):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    "true_class": ["negative", "neutral", "positive"][i],
                    "pred_class": ["negative", "neutral", "positive"][j],
                    "count": int(cm[i, j]),
                    "percentage": float(cm[i, j] / np.sum(cm[i]) * 100)
                })
    
    # Sort by count descending
    confusion_pairs.sort(key=lambda x: x["count"], reverse=True)
    
    return {
        "n": int(len(df)),
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "macro_metrics": {
            "precision": float(p),
            "recall": float(r),
            "f1_score": float(f1)
        },
        "micro_metrics": {
            "precision": float(p_micro),
            "recall": float(r_micro), 
            "f1_score": float(f1_micro)
        },
        "weighted_metrics": {
            "precision": float(p_weighted),
            "recall": float(r_weighted),
            "f1_score": float(f1_weighted)
        },
        "statistical_metrics": {
            "matthews_correlation_coefficient": mcc,
            "cohens_kappa": kappa
        },
        "auc_metrics": {
            "per_class_roc_auc": auc_scores,
            "macro_roc_auc": macro_auc,
            "per_class_pr_auc": ap_scores,
            "macro_pr_auc": macro_ap
        },
        "confusion_matrix": cm.tolist(),
        "class_distribution": class_distribution,
        "top_confusion_pairs": confusion_pairs[:5],  # Top 5 most confused pairs
        "per_class": {
            "negative": {
                "precision": float(p_c[0]), 
                "recall": float(r_c[0]), 
                "f1": float(f1_c[0]),
                "support": int(class_counts[0])
            },
            "neutral": {
                "precision": float(p_c[1]), 
                "recall": float(r_c[1]), 
                "f1": float(f1_c[1]),
                "support": int(class_counts[1])
            },
            "positive": {
                "precision": float(p_c[2]), 
                "recall": float(r_c[2]), 
                "f1": float(f1_c[2]),
                "support": int(class_counts[2])
            }
        },
        "model_performance_summary": {
            "overall_quality": "excellent" if acc > 0.9 else "good" if acc > 0.8 else "fair" if acc > 0.7 else "poor",
            "class_balance": "balanced" if min(class_counts) / max(class_counts) > 0.5 else "imbalanced",
            "key_strengths": [],
            "areas_for_improvement": []
        }
    }


def embed_for_clustering(df: pd.DataFrame, text_col: str, model_name="sentence-transformers/all-MiniLM-L6-v2", sample: int=20000) -> np.ndarray:
    if SentenceTransformer is None:
        raise SystemExit("sentence-transformers not installed. `pip install sentence-transformers`")
    texts = df[text_col].astype(str).tolist()
    if sample and len(texts) > sample:
        texts = texts[:sample]
    model = SentenceTransformer(model_name)
    X = model.encode(texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return X


def eval_clustering(clean_path: Path, clusters_path: Path, sample:int=20000, model_name="sentence-transformers/all-MiniLM-L6-v2") -> Dict:
    df = load_table(clean_path)
    if "meta_cluster" not in load_table(clusters_path).columns:
        raise SystemExit("Cluster parquet must contain 'meta_cluster' column.")

    clus = load_table(clusters_path)
    if len(clus) != len(df):
        raise SystemExit("Cluster file length must match clean file length for evaluation.")

    df["meta_cluster"] = clus["meta_cluster"].values
    text_col = resolve_text_col(df)

    X = embed_for_clustering(df, text_col=text_col, model_name=model_name, sample=sample)
    y = df["meta_cluster"].values[:len(X)]

    # Core clustering metrics
    sil = float(silhouette_score(X, y))
    ch_score = float(calinski_harabasz_score(X, y))
    db_score = float(davies_bouldin_score(X, y))
    
    # Cluster statistics
    unique_clusters = np.unique(y)
    n_clusters = len(unique_clusters)
    cluster_sizes = [int(np.sum(y == cluster)) for cluster in unique_clusters]
    
    # Cluster balance analysis
    min_size, max_size = min(cluster_sizes), max(cluster_sizes)
    cluster_balance_ratio = min_size / max_size if max_size > 0 else 0
    
    # Inertia calculation (within-cluster sum of squares)
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X)
        inertia = float(kmeans.inertia_)
        
        # Calculate between-cluster sum of squares for additional insight
        total_ss = np.sum((X - np.mean(X, axis=0)) ** 2)
        between_ss = total_ss - inertia
        explained_variance_ratio = between_ss / total_ss if total_ss > 0 else 0
    except Exception:
        inertia = float('nan')
        explained_variance_ratio = float('nan')
    
    out = {
        "n": int(len(X)),
        "n_clusters": n_clusters,
        "internal_metrics": {
            "silhouette_score": sil,
            "calinski_harabasz_score": ch_score,
            "davies_bouldin_score": db_score,
            "inertia": inertia,
            "explained_variance_ratio": float(explained_variance_ratio)
        },
        "cluster_statistics": {
            "cluster_sizes": cluster_sizes,
            "min_cluster_size": min_size,
            "max_cluster_size": max_size,
            "mean_cluster_size": float(np.mean(cluster_sizes)),
            "std_cluster_size": float(np.std(cluster_sizes)),
            "cluster_balance_ratio": float(cluster_balance_ratio)
        },
        "quality_assessment": {
            "silhouette_quality": (
                "excellent" if sil > 0.7 else
                "good" if sil > 0.5 else
                "fair" if sil > 0.25 else
                "poor"
            ),
            "cluster_balance": (
                "well_balanced" if cluster_balance_ratio > 0.5 else
                "moderately_balanced" if cluster_balance_ratio > 0.2 else
                "imbalanced"
            )
        }
    }

    # Optional: ARI vs. original categories if present (external agreement)
    if "category" in df.columns:
        # Map sparse categories to integers
        cats = df["category"].astype(str).values[:len(X)]
        # If categories are very unique per-row, ARI is meaningless. Guard by checking cardinality.
        if len(set(cats)) < len(cats) * 0.5:
            ari_score = float(adjusted_rand_score(cats, y))
            out["external_validation"] = {
                "adjusted_rand_index_vs_raw_category": ari_score,
                "ari_interpretation": (
                    "excellent_agreement" if ari_score > 0.9 else
                    "substantial_agreement" if ari_score > 0.8 else
                    "moderate_agreement" if ari_score > 0.6 else
                    "fair_agreement" if ari_score > 0.4 else
                    "poor_agreement"
                ),
                "n_original_categories": len(set(cats))
            }
        else:
            out["external_validation"] = {
                "adjusted_rand_index_vs_raw_category": float('nan'),
                "note": "Too many unique categories for meaningful ARI comparison",
                "n_original_categories": len(set(cats))
            }
    
    return out


def eval_generation(generated_dir: Path, reference_dir: Optional[Path]) -> Dict:
    if rouge_scorer is None or sacrebleu is None:
        raise SystemExit("Install rouge-score and sacrebleu for generative evaluation.")

    gen_files = sorted([p for p in Path(generated_dir).glob("*.md")])
    if not gen_files:
        return {"n": 0}

    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge1, rougel, bleu_scores = [], [], []

    if reference_dir and Path(reference_dir).exists():
        refs = {p.stem: p.read_text() for p in Path(reference_dir).glob("*.md")}
    else:
        refs = {}

    for gf in gen_files:
        hyp = gf.read_text()
        key = gf.stem
        if key in refs:
            ref = refs[key]
            rs = rouge.score(ref, hyp)
            rouge1.append(rs["rouge1"].fmeasure)
            rougel.append(rs["rougeL"].fmeasure)
            bleu = sacrebleu.corpus_bleu([hyp], [[ref]]).score
            bleu_scores.append(bleu)

    # Content analysis for all generated texts
    all_texts = [p.read_text() for p in gen_files]
    word_counts = [len(text.split()) for text in all_texts]
    sentence_counts = [len([s for s in text.split('.') if s.strip()]) for text in all_texts]
    char_counts = [len(text) for text in all_texts]
    
    # Calculate readability and content metrics
    avg_words_per_sentence = [wc/sc if sc > 0 else 0 for wc, sc in zip(word_counts, sentence_counts)]
    
    # Lexical diversity (type-token ratio)
    lexical_diversity = []
    for text in all_texts:
        words = text.lower().split()
        unique_words = set(words)
        ttr = len(unique_words) / len(words) if len(words) > 0 else 0
        lexical_diversity.append(ttr)
    
    # Content structure analysis
    structure_scores = []
    for text in all_texts:
        # Check for common buyer's guide elements
        structure_elements = {
            'has_recommendations': any(keyword in text.lower() for keyword in ['recommend', 'best', 'top', 'choice']),
            'has_comparisons': any(keyword in text.lower() for keyword in ['vs', 'versus', 'compared', 'difference']),
            'has_pros_cons': any(keyword in text.lower() for keyword in ['pros', 'cons', 'advantages', 'disadvantages']),
            'has_price_info': any(keyword in text.lower() for keyword in ['price', 'cost', '$', 'budget', 'expensive']),
            'has_ratings': any(keyword in text.lower() for keyword in ['rating', 'stars', 'score', '/5', 'review'])
        }
        structure_score = sum(structure_elements.values()) / len(structure_elements)
        structure_scores.append(structure_score)
    
    content_metrics = {
        "text_statistics": {
            "avg_words": float(np.mean(word_counts)),
            "std_words": float(np.std(word_counts)),
            "min_words": int(min(word_counts)),
            "max_words": int(max(word_counts)),
            "avg_sentences": float(np.mean(sentence_counts)),
            "avg_chars": float(np.mean(char_counts)),
            "avg_words_per_sentence": float(np.mean(avg_words_per_sentence))
        },
        "content_quality": {
            "avg_lexical_diversity": float(np.mean(lexical_diversity)),
            "avg_structure_score": float(np.mean(structure_scores)),
            "consistency": {
                "word_count_cv": float(np.std(word_counts) / np.mean(word_counts)) if np.mean(word_counts) > 0 else 0,
                "lexical_diversity_cv": float(np.std(lexical_diversity) / np.mean(lexical_diversity)) if np.mean(lexical_diversity) > 0 else 0
            }
        },
        "quality_assessment": {
            "length_consistency": "good" if np.std(word_counts) / np.mean(word_counts) < 0.3 else "variable",
            "structural_completeness": "excellent" if np.mean(structure_scores) > 0.8 else "good" if np.mean(structure_scores) > 0.6 else "fair",
            "lexical_richness": "high" if np.mean(lexical_diversity) > 0.6 else "medium" if np.mean(lexical_diversity) > 0.4 else "low"
        }
    }

    if not refs:
        # No references → return comprehensive content analysis
        return {
            "n": len(gen_files), 
            **content_metrics,
            "note": "No references provided; ROUGE/BLEU skipped."
        }

    # With references - add similarity metrics
    similarity_metrics = {
        "rouge_metrics": {
            "rouge1_f": float(np.mean(rouge1)) if rouge1 else None,
            "rouge1_std": float(np.std(rouge1)) if rouge1 else None,
            "rougeL_f": float(np.mean(rougel)) if rougel else None,
            "rougeL_std": float(np.std(rougel)) if rougel else None
        },
        "bleu_metrics": {
            "bleu_score": float(np.mean(bleu_scores)) if bleu_scores else None,
            "bleu_std": float(np.std(bleu_scores)) if bleu_scores else None
        },
        "similarity_assessment": {
            "rouge_quality": (
                "excellent" if np.mean(rouge1) > 0.6 else
                "good" if np.mean(rouge1) > 0.4 else
                "fair" if np.mean(rouge1) > 0.2 else
                "poor"
            ) if rouge1 else "no_data"
        }
    }

    return {
        "n": len(rouge1),
        **content_metrics,
        **similarity_metrics
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate classifier, clustering, and generation.")
    ap.add_argument("--clean", default="artifacts/clean_reviews.parquet")
    ap.add_argument("--clusters", default="artifacts/cluster_assignments_product_based.parquet")
    ap.add_argument("--clf_dir", default="artifacts/clf")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--sample_cls", type=int, default=20000)
    ap.add_argument("--sample_cluster", type=int, default=20000)
    ap.add_argument("--gen_dir", default="artifacts/summaries")
    ap.add_argument("--ref_dir", default=None, help="Optional directory with reference .md files for ROUGE/BLEU")
    ap.add_argument("--out", default="artifacts/eval_report.json")
    return ap.parse_args()


def main():
    args = parse_args()

    # Run individual evaluations
    classifier_results = eval_classifier(Path(args.clean), Path(args.clf_dir), device_str=args.device, sample=args.sample_cls)
    clustering_results = eval_clustering(Path(args.clean), Path(args.clusters), sample=args.sample_cluster)
    generation_results = eval_generation(Path(args.gen_dir), Path(args.ref_dir) if args.ref_dir else None)
    
    # Create comprehensive pipeline evaluation summary
    pipeline_summary = {
        "overall_pipeline_health": "healthy",
        "key_findings": [],
        "recommendations": [],
        "scores": {}
    }
    
    # Analyze classifier performance
    clf_acc = classifier_results.get('accuracy', 0)
    clf_f1 = classifier_results.get('macro_metrics', {}).get('f1_score', 0)
    pipeline_summary["scores"]["classifier_accuracy"] = clf_acc
    pipeline_summary["scores"]["classifier_f1_macro"] = clf_f1
    
    if clf_acc > 0.85:
        pipeline_summary["key_findings"].append("Sentiment classifier shows excellent performance")
    elif clf_acc > 0.75:
        pipeline_summary["key_findings"].append("Sentiment classifier shows good performance")
    else:
        pipeline_summary["key_findings"].append("Sentiment classifier needs improvement")
        pipeline_summary["recommendations"].append("Consider retraining sentiment classifier with more data or different architecture")
    
    # Check class balance in classifier
    pos_f1 = classifier_results.get('per_class', {}).get('positive', {}).get('f1', 0)
    if pos_f1 > 0.8:
        pipeline_summary["key_findings"].append("Positive sentiment detection is strong (critical for buyer's guides)")
    else:
        pipeline_summary["recommendations"].append("Improve positive sentiment detection for better product recommendations")
    
    # Analyze clustering quality
    sil_score = clustering_results.get('internal_metrics', {}).get('silhouette_score', 0)
    pipeline_summary["scores"]["clustering_silhouette"] = sil_score
    
    if sil_score > 0.5:
        pipeline_summary["key_findings"].append("Product clustering shows good separation")
    elif sil_score > 0.25:
        pipeline_summary["key_findings"].append("Product clustering shows moderate quality")
    else:
        pipeline_summary["key_findings"].append("Product clustering needs improvement")
        pipeline_summary["recommendations"].append("Consider different clustering parameters or features")
    
    # Check cluster balance
    balance_ratio = clustering_results.get('cluster_statistics', {}).get('cluster_balance_ratio', 0)
    if balance_ratio < 0.3:
        pipeline_summary["recommendations"].append("Clusters are imbalanced - consider rebalancing strategy")
    
    # Analyze generation quality
    n_summaries = generation_results.get('n', 0)
    avg_words = generation_results.get('text_statistics', {}).get('avg_words', 0)
    pipeline_summary["scores"]["n_summaries_generated"] = n_summaries
    pipeline_summary["scores"]["avg_summary_length"] = avg_words
    
    if n_summaries > 0:
        if avg_words > 50 and avg_words < 200:
            pipeline_summary["key_findings"].append("Generated summaries have appropriate length for buyer's guides")
        elif avg_words < 50:
            pipeline_summary["recommendations"].append("Increase summary length for more comprehensive buyer's guides")
        else:
            pipeline_summary["recommendations"].append("Consider reducing summary length for better readability")
            
        structure_score = generation_results.get('content_quality', {}).get('avg_structure_score', 0)
        if structure_score > 0.7:
            pipeline_summary["key_findings"].append("Generated content has good buyer's guide structure")
        else:
            pipeline_summary["recommendations"].append("Improve prompt engineering for better buyer's guide structure")
    
    # Overall health assessment
    health_scores = [
        min(clf_acc * 1.2, 1.0),  # Weight classifier performance highly
        min(sil_score * 2, 1.0),  # Normalize silhouette score
        min((n_summaries / 20), 1.0) if n_summaries > 0 else 0  # Generation coverage
    ]
    
    overall_health = np.mean(health_scores)
    if overall_health > 0.8:
        pipeline_summary["overall_pipeline_health"] = "excellent"
    elif overall_health > 0.6:
        pipeline_summary["overall_pipeline_health"] = "good"
    elif overall_health > 0.4:
        pipeline_summary["overall_pipeline_health"] = "fair"
    else:
        pipeline_summary["overall_pipeline_health"] = "needs_improvement"
    
    pipeline_summary["scores"]["overall_health_score"] = float(overall_health)
    
    # Add timestamp and metadata
    from datetime import datetime
    pipeline_summary["evaluation_metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "samples_used": {
            "classifier": args.sample_cls,
            "clustering": args.sample_cluster
        },
        "device": args.device
    }
    
    # Final report structure
    report = {
        "pipeline_summary": pipeline_summary,
        "classifier": classifier_results,
        "clustering": clustering_results,
        "generation": generation_results
    }

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
