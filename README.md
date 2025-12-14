# 🤖 RoboReviews - AI Buyer's Guide Generator# Automated-Customer-Reviews

second-last-project

Generate comprehensive, data-driven buyer's guides from customer review data using AI.

## ✨ Features

- **Multi-Model Support**: Choose between 3 AI models (Flan-T5, Ollama Qwen, Raw Qwen)
- **Smart Analysis**: Processes thousands of customer reviews to extract insights
- **Beautiful UI**: Clean Gradio interface with styled recommendation cards
- **Sentiment Analysis**: Identifies top products and common complaints
- **Production Ready**: Ollama integration provides 20x speed improvement over raw models

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Ollama (recommended for best performance)

### Installation

1. **Install Ollama** (for best performance):
```bash
brew install ollama
ollama serve &
ollama pull qwen2.5:7b
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the app**:
```bash
python src/app_unified.py
```

4. **Open in browser**: http://127.0.0.1:7860

## 📊 Model Comparison

| Model | Speed | Quality | Memory | Best For |
|-------|-------|---------|--------|----------|
| **Ollama Qwen 2.5 7B** ⭐ | ⚡⚡⚡ 2-5 tok/s | ★★★★★ | 6GB | **Production (recommended)** |
| Flan-T5-large | ⚡⚡⚡⚡ 10-20 tok/s | ★★★★☆ | 2GB | Fast prototyping |
| Raw Qwen 2.5 7B | 🐌 0.1 tok/s | ★★★★★ | 16GB | Only with NVIDIA GPU |

**Time estimates for 1,500 token guide:**
- Ollama: 5-12 minutes ⚡
- Flan-T5: 1-2 minutes ⚡⚡
- Raw Qwen: 4+ hours 🐌 (not recommended on Mac)

## 🎯 Usage

1. Select a model backend (Ollama recommended)
2. Choose a product category from the dropdown
3. Adjust generation parameters (optional)
4. Click "Generate Buyer's Guide"
5. Review the top 3 recommendations and complete guide

## 📁 Project Structure

```
Automated-Customer-Reviews/
├── src/
│   ├── app_unified.py          # Main application (USE THIS)
│   ├── preprocess.py            # Data preprocessing
│   ├── train_classifier.py     # Sentiment classifier training
│   ├── evaluate_models.py      # Model evaluation
│   ├── generate_summaries.py   # Batch summary generation
│   └── constants.py            # Configuration constants
├── prompts/
│   └── examples_v3.json        # Current prompt templates
├── artifacts/
│   ├── clean_reviews.parquet            # Preprocessed reviews
│   ├── cluster_assignments_optimized.parquet  # Product categories
│   ├── pred_labels.parquet              # Sentiment predictions
│   └── summaries/                       # Pre-generated summaries
├── data/                        # Raw review data
├── notebooks/                   # Analysis notebooks
├── old_apps/                    # Archived old versions
├── old_docs/                    # Archived documentation
└── old_prompts/                 # Archived prompt versions
```

## 🔧 Configuration

Edit `src/constants.py` to customize:
- Model paths
- Generation parameters
- Data paths
- UI settings

## 📝 Output Format

Generated guides include:

1. **🏆 Top 3 Recommendations**
   - #1 Highest Rated
   - #2 Best Reviewed  
   - #3 Most Popular

2. **🔍 When to Choose Each**
   - Specific use cases for each product

3. **⚠️ What Customers Complain About**
   - Common issues by product

4. **❌ Product to Avoid**
   - Lowest rated option with explanation

5. **The Bottom Line**
   - Final recommendation summary

## 🛠️ Development

### Data Pipeline

```bash
# 1. Preprocess reviews
python src/preprocess.py

# 2. Train sentiment classifier
python src/train_classifier.py

# 3. Generate summaries (batch)
python src/generate_summaries.py

# 4. Run web app
python src/app_unified.py
```

### Troubleshooting

**Ollama not running:**
```bash
ollama serve &
```

**Model not found:**
```bash
ollama pull qwen2.5:7b
```

**Port already in use:**
```bash
pkill -f "python.*app"
python src/app_unified.py
```

## 📦 Requirements

- pandas
- numpy
- transformers
- torch
- gradio
- requests
- scikit-learn

See `requirements.txt` for complete list.

## 🎓 Dataset

Amazon product reviews dataset with:
- 28,000+ reviews analyzed
- 12+ product categories
- Sentiment classification
- Category clustering

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Hugging Face Transformers
- Ollama
- Gradio
- Amazon Customer Reviews Dataset
