# 📚 Project Files Reference

Quick reference for all active files in the cleaned project.

## 🎯 Main Application

### `src/app_unified.py` ⭐ **START HERE**
**Purpose**: Main web application with unified model interface  
**What it does**:
- Runs Gradio web interface on port 7860
- Provides model selection (Flan-T5, Ollama, Raw Qwen)
- Generates buyer's guides from review data
- Parses and displays top 3 recommendations in styled boxes
- Includes all functionality from previous separate apps

**How to run**:
```bash
python src/app_unified.py
```

**Dependencies**: 
- Ollama server (for Ollama backend)
- artifacts/clean_reviews.parquet
- artifacts/cluster_assignments_optimized.parquet
- artifacts/pred_labels.parquet
- prompts/examples_v3.json

---

## 🔧 Data Pipeline Scripts

### `src/preprocess.py`
**Purpose**: Clean and prepare raw review data  
**What it does**:
- Loads raw CSV files from `data/`
- Cleans text (removes HTML, special chars)
- Extracts features (price, dates, stars)
- Saves to `artifacts/clean_reviews.parquet`

**How to run**:
```bash
python src/preprocess.py
```

### `src/train_classifier.py`
**Purpose**: Train sentiment classification model  
**What it does**:
- Trains DistilBERT for sentiment analysis
- Fine-tunes on review data
- Saves model to `artifacts/clf/`
- Generates predictions in `artifacts/pred_labels.parquet`

**How to run**:
```bash
python src/train_classifier.py
```

### `src/generate_summaries.py`
**Purpose**: Batch generate buyer's guides  
**What it does**:
- Generates guides for all categories
- Uses Flan-T5 or Qwen models
- Saves markdown files to `artifacts/summaries/`
- Can be used to pre-generate content

**How to run**:
```bash
python src/generate_summaries.py --limit_clusters 5
```

### `src/evaluate_models.py`
**Purpose**: Evaluate classifier performance  
**What it does**:
- Tests sentiment classifier accuracy
- Generates metrics (F1, precision, recall)
- Saves results to `artifacts/eval/`

**How to run**:
```bash
python src/evaluate_models.py
```

### `src/constants.py`
**Purpose**: Configuration constants  
**What it contains**:
- File paths
- Model settings
- Default parameters
- Cluster definitions

---

## 📝 Prompt Templates

### `prompts/examples_v3.json` ⭐
**Purpose**: Current prompt examples for guide generation  
**What it contains**:
- System instruction defining output format
- Example buyer's guides
- Format: #1/#2/#3 recommendations
- Includes complaints section
- Used by all models

**Structure**:
```json
{
  "instruction": "Generate buyer's guide...",
  "examples": [
    {
      "input": "Facts about products...",
      "output": "## 🏆 Top 3 Recommendations..."
    }
  ]
}
```

---

## 📊 Data Files

### `artifacts/clean_reviews.parquet`
**Created by**: `preprocess.py`  
**Contains**: ~28,000 cleaned Amazon product reviews  
**Columns**: product_name, text_clean, stars, price, review_date, brand_name

### `artifacts/cluster_assignments_optimized.parquet`
**Created by**: Manual clustering (archived scripts)  
**Contains**: Product category assignments  
**Column**: meta_cluster_name (12 categories)

### `artifacts/pred_labels.parquet`
**Created by**: `train_classifier.py`  
**Contains**: Sentiment predictions (positive/neutral/negative)  
**Column**: pred_label

### `artifacts/summaries/`
**Created by**: `generate_summaries.py` or `app_unified.py`  
**Contains**: Pre-generated buyer's guide markdown files  
**Format**: `{category_name}.md`

---

## 📖 Documentation

### `README.md` ⭐
**Purpose**: Main project documentation  
**Sections**:
- Quick start guide
- Model comparison
- Installation instructions
- Usage guide
- Project structure
- Troubleshooting

### `CLEANUP_SUMMARY.md`
**Purpose**: Documents the cleanup process  
**Sections**:
- What was removed
- What was archived
- Before/after structure
- Clean structure overview

### `FILE_REFERENCE.md` (this file)
**Purpose**: Quick reference for all project files  
**Use**: To understand what each file does

---

## 🗂️ Archive Folders (Optional)

### `old_apps/`
**Contains**: 8 previous app versions  
**Can delete**: Yes, once unified app is confirmed working

### `old_docs/`
**Contains**: 13 previous documentation files  
**Can delete**: Yes, kept for historical reference only

### `old_prompts/`
**Contains**: 5 previous prompt versions  
**Can delete**: Yes, only examples_v3.json is used now

---

## 🚀 Quick Commands

**Run the app**:
```bash
python src/app_unified.py
```

**Regenerate data pipeline**:
```bash
python src/preprocess.py
python src/train_classifier.py
python src/generate_summaries.py
```

**Check app status**:
```bash
ps aux | grep app_unified
```

**Restart app**:
```bash
pkill -f "python.*app_unified"
python src/app_unified.py
```

**Start Ollama**:
```bash
ollama serve &
```

---

## 📦 Dependencies

See `requirements.txt` for complete list:
- **gradio** - Web interface
- **transformers** - LLM models
- **torch** - Deep learning
- **pandas** - Data manipulation
- **scikit-learn** - ML utilities
- **requests** - Ollama HTTP API

---

## 💡 Tips

1. **Always use** `app_unified.py` - it's the only app you need
2. **Only** `examples_v3.json` is used - ignore old prompts
3. **Archive folders** can be deleted once confident
4. **README.md** has comprehensive guides
5. **Ollama** is recommended for best results
