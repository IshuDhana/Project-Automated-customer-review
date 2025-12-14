#!/usr/bin/env python3
"""
Unified RoboReviews app with multiple model backends:
- Flan-T5-large (fast, good quality)
- Ollama/Qwen 2.5 7B (best quality, 10-20x faster than raw Qwen)
- Raw Qwen 2.5 7B (highest quality but slow on Mac)

All functionalities in one clean interface.
"""
import sys
sys.path.append('src')

from app import (
    DEFAULT_CLEAN, DEFAULT_CLUSTERS, DEFAULT_LABELS,
    load_table, aggregate_products, pick_text_col,
    build_facts_block, load_nshots, build_prompt, SYSTEM_INSTR,
    load_llm, generate_text
)

import pandas as pd
import gradio as gr
import requests
import time
from typing import Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ModelBackend:
    """Abstract base for different model backends."""
    
    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Tuple[str, dict]:
        """Generate text and return (output, stats)."""
        raise NotImplementedError


class FlanT5Backend(ModelBackend):
    """Flan-T5-large backend (fast, good quality)."""
    
    def __init__(self, device='cpu'):
        print("Loading Flan-T5-large...")
        self.tokenizer, self.model, self.device = load_llm('google/flan-t5-large', device, 'auto')
        print("✅ Flan-T5-large loaded")
    
    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Tuple[str, dict]:
        start = time.time()
        output = generate_text(
            self.tokenizer, 
            self.model, 
            self.device, 
            prompt, 
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0
        )
        elapsed = time.time() - start
        
        stats = {
            'time': elapsed,
            'tokens': len(output.split()) * 1.3,  # Rough estimate
            'tokens_per_sec': (len(output.split()) * 1.3) / elapsed if elapsed > 0 else 0
        }
        
        return output, stats


class OllamaBackend(ModelBackend):
    """Ollama backend (quantized, Apple Silicon optimized)."""
    
    def __init__(self, model_name='qwen2.5:7b'):
        self.model_name = model_name
        # Check if Ollama is running
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if resp.status_code == 200:
                print(f"✅ Ollama connected ({model_name})")
            else:
                raise ConnectionError("Ollama server error")
        except:
            raise RuntimeError(
                "❌ Ollama server not running!\n"
                "Start it with: ollama serve &\n"
                "Then: ollama pull qwen2.5:7b"
            )
    
    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Tuple[str, dict]:
        start = time.time()
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    }
                },
                timeout=600  # 10 minute timeout
            )
            response.raise_for_status()
            output = response.json()["response"]
            
            elapsed = time.time() - start
            stats = {
                'time': elapsed,
                'tokens': len(output.split()) * 1.3,
                'tokens_per_sec': (len(output.split()) * 1.3) / elapsed if elapsed > 0 else 0
            }
            
            return output, stats
            
        except Exception as e:
            return f"❌ Ollama error: {str(e)}", {'time': 0, 'tokens': 0, 'tokens_per_sec': 0}


class QwenBackend(ModelBackend):
    """Raw Qwen 2.5 7B backend (slow on Mac without CUDA)."""
    
    def __init__(self, device='mps'):
        print("Loading Qwen 2.5 7B (this may take a minute)...")
        print("⚠️  Warning: Raw Qwen is very slow on Apple Silicon (0.1 tokens/sec)")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            trust_remote_code=True
        )
        
        # Load in float16 (no quantization available on Mac)
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        self.device = device
        print("✅ Qwen loaded (float16, no quantization)")
    
    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Tuple[str, dict]:
        start = time.time()
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device != 'cpu':
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        output = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        elapsed = time.time() - start
        stats = {
            'time': elapsed,
            'tokens': len(output.split()) * 1.3,
            'tokens_per_sec': (len(output.split()) * 1.3) / elapsed if elapsed > 0 else 0
        }
        
        return output, stats


def generate_summary_unified(
    cluster_name: str,
    df: pd.DataFrame,
    backend: ModelBackend,
    text_col: str = "text_clean",
    shots_path: str = "prompts/examples_v3.json",
    top_n: int = 3,
    min_reviews: int = 50,
    max_new_tokens: int = 1500,
    temperature: float = 0.7
) -> Tuple[str, dict]:
    """
    Generate buyer's guide using specified backend.
    
    Returns:
        (generated_text, stats_dict)
    """
    g = df[df['meta_cluster_name'].astype(str) == str(cluster_name)].copy()
    
    if g.empty:
        return f"❌ No data for cluster '{cluster_name}'", {}
    
    # Aggregate products
    agg = aggregate_products(g, text_col=text_col)
    
    # Filter by minimum reviews
    agg = agg[agg['n_reviews'] >= min_reviews]
    
    if agg.empty:
        return f"❌ No products with >={min_reviews} reviews in '{cluster_name}'", {}
    
    # Sort by stars descending, take top N
    topk = agg.sort_values('avg_stars', ascending=False).head(top_n)
    
    if topk.empty:
        return f"❌ Could not find top products for '{cluster_name}'", {}
    
    # Build facts
    facts = build_facts_block(cluster_name, topk, df_full=g)
    
    # Load examples and build prompt
    shots = load_nshots(shots_path)
    prompt = build_prompt(shots, facts)
    
    # Generate with selected backend
    print(f"\n{'='*80}")
    print(f"Generating guide for: {cluster_name}")
    print(f"Backend: {backend.__class__.__name__}")
    print(f"Top products: {len(topk)}")
    print(f"{'='*80}\n")
    
    output, stats = backend.generate(prompt, max_new_tokens, temperature)
    
    print(f"\n✅ Generated in {stats.get('time', 0):.1f}s (~{stats.get('tokens_per_sec', 0):.1f} tokens/sec)")
    print(f"📝 Output length: {len(output)} chars\n")
    
    return output, stats


def build_unified_app():
    """Build unified Gradio interface with model selection."""
    
    def parse_recommendations(text):
        """Parse the generated text to extract top 3 recommendations."""
        recs = ["", "", ""]
        
        # Look for the top 3 recommendations section
        lines = text.split('\n')
        rec_idx = -1
        current_rec = ""
        
        for line in lines:
            # Detect start of recommendation sections
            if '#1' in line or '🥇' in line:
                if current_rec and rec_idx >= 0:
                    recs[rec_idx] = current_rec.strip()
                rec_idx = 0
                current_rec = f"## 🥇 {line.split('#1')[-1].strip()}\n"
            elif '#2' in line or '🥈' in line:
                if current_rec and rec_idx >= 0:
                    recs[rec_idx] = current_rec.strip()
                rec_idx = 1
                current_rec = f"## 🥈 {line.split('#2')[-1].strip()}\n"
            elif '#3' in line or '🥉' in line:
                if current_rec and rec_idx >= 0:
                    recs[rec_idx] = current_rec.strip()
                rec_idx = 2
                current_rec = f"## 🥉 {line.split('#3')[-1].strip()}\n"
            elif rec_idx >= 0 and rec_idx < 3:
                # Stop if we hit another major section
                if line.startswith('## ') and not any(x in line for x in ['#1', '#2', '#3', '🥇', '🥈', '🥉']):
                    if current_rec:
                        recs[rec_idx] = current_rec.strip()
                    break
                else:
                    current_rec += line + "\n"
        
        # Save last recommendation
        if rec_idx >= 0 and rec_idx < 3 and current_rec:
            recs[rec_idx] = current_rec.strip()
        
        return recs[0], recs[1], recs[2]
    
    # Load data once at startup
    df = load_table(DEFAULT_CLEAN)
    clusters = load_table(DEFAULT_CLUSTERS)
    labels = load_table(DEFAULT_LABELS)
    
    # Merge data
    df['meta_cluster_name'] = clusters['meta_cluster_name'].values
    
    # Handle label mapping
    if 'pred_label' in labels.columns:
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        df['label'] = labels['pred_label'].map(label_map).fillna(1).astype(int).values
    else:
        df['label'] = labels['label'].astype(int).values
    
    # Get available clusters
    available_clusters = sorted(df['meta_cluster_name'].dropna().unique())
    
    # Model backends (lazy loading)
    backends = {}
    
    def get_backend(model_choice: str) -> ModelBackend:
        """Get or create backend instance."""
        if model_choice not in backends:
            if model_choice == "Flan-T5-large (Fast)":
                backends[model_choice] = FlanT5Backend(device='cpu')
            elif model_choice == "Ollama Qwen 2.5 7B (Best)":
                backends[model_choice] = OllamaBackend(model_name='qwen2.5:7b')
            elif model_choice == "Raw Qwen 2.5 7B (Slow)":
                backends[model_choice] = QwenBackend(device='mps')
        return backends[model_choice]
    
    # Custom CSS for better design
    custom_css = """
    .recommendation-card {
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .card-gold {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border: 3px solid #F59E0B;
    }
    
    .card-silver {
        background: linear-gradient(135deg, #F3F4F6 0%, #E5E7EB 100%);
        border: 3px solid #9CA3AF;
    }
    
    .card-bronze {
        background: linear-gradient(135deg, #FED7AA 0%, #FDBA74 100%);
        border: 3px solid #EA580C;
    }
    
    .product-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 12px;
        color: #1F2937;
    }
    
    .product-stats {
        font-size: 0.95rem;
        font-weight: 600;
        color: #4B5563;
        margin-bottom: 8px;
    }
    
    .product-description {
        line-height: 1.6;
        color: #374151;
    }
    """
    
    # Build interface
    with gr.Blocks(title="🤖 RoboReviews - Unified", css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 RoboReviews - AI Buyer's Guide Generator
        
        Generate comprehensive buyer's guides from customer review data. **Choose your model** for different speed/quality tradeoffs.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")
                
                model_choice = gr.Radio(
                    choices=[
                        "Flan-T5-large (Fast)",
                        "Ollama Qwen 2.5 7B (Best)",
                        "Raw Qwen 2.5 7B (Slow)"
                    ],
                    value="Ollama Qwen 2.5 7B (Best)",
                    label="🤖 Model Backend",
                    info="Ollama recommended for best quality+speed"
                )
                
                cluster_dropdown = gr.Dropdown(
                    choices=available_clusters,
                    label="📦 Product Category",
                    value=available_clusters[0] if available_clusters else None
                )
                
                with gr.Row():
                    top_n = gr.Slider(1, 10, value=3, step=1, label="Top N Products")
                    min_reviews = gr.Slider(10, 200, value=50, step=10, label="Min Reviews")
                
                with gr.Row():
                    max_tokens = gr.Slider(500, 2000, value=1500, step=100, label="Max Tokens")
                    temperature = gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="Temperature")
                
                shots_path = gr.Textbox(
                    value="prompts/examples_v3.json",
                    label="📝 Examples File"
                )
                
                generate_btn = gr.Button("🎯 Generate Buyer's Guide", variant="primary", size="lg")
                
                # Stats display
                with gr.Accordion("📊 Generation Stats", open=False):
                    stats_md = gr.Markdown("")
            
            with gr.Column(scale=2):
                gr.Markdown("### 🏆 Our Top 3 Recommendations")
                
                # Top 3 recommendation boxes
                with gr.Row():
                    rec1 = gr.Markdown("", elem_classes=["recommendation-card", "card-gold"])
                    rec2 = gr.Markdown("", elem_classes=["recommendation-card", "card-silver"])
                    rec3 = gr.Markdown("", elem_classes=["recommendation-card", "card-bronze"])
                
                gr.Markdown("---")
                gr.Markdown("### 📄 Complete Guide")
                output_md = gr.Markdown(label="Output", elem_classes=["guide-output"])
        
        # Model comparison table
        with gr.Accordion("ℹ️ Model Comparison", open=False):
            gr.Markdown("""
            | Model | Speed | Quality | Memory | Best For |
            |-------|-------|---------|--------|----------|
            | **Ollama Qwen 2.5 7B** | ⚡⚡⚡ 2-5 tok/s | ★★★★★ Excellent | 6GB | **Production (recommended)** |
            | Flan-T5-large | ⚡⚡⚡⚡ 10-20 tok/s | ★★★★☆ Very Good | 2GB | Fast prototyping |
            | Raw Qwen 2.5 7B | 🐌 0.1 tok/s | ★★★★★ Excellent | 16GB | Only if you have NVIDIA GPU |
            
            **Time estimates for 1,500 token guide:**
            - Ollama: 5-12 minutes ⚡
            - Flan-T5: 1-2 minutes ⚡⚡
            - Raw Qwen: 4+ hours 🐌 (not recommended on Mac)
            """)
        
        # Dataset stats
        with gr.Accordion("📈 Dataset Info", open=False):
            gr.Markdown(f"""
            **Dataset Statistics:**
            - Total reviews: **{len(df):,}**
            - Product categories: **{len(available_clusters)}**
            - Unique products: **{df['product_name'].nunique():,}**
            - Date range: {df['review_date'].min() if 'review_date' in df.columns else 'N/A'} to {df['review_date'].max() if 'review_date' in df.columns else 'N/A'}
            
            **Available Categories:**
            {', '.join(available_clusters[:12])}...
            """)
        
        # Generate function
        def generate_wrapper(model, cluster, top_n_val, min_rev, max_tok, temp, shots):
            try:
                # Get backend
                backend = get_backend(model)
                
                # Generate
                result, stats = generate_summary_unified(
                    cluster_name=cluster,
                    df=df,
                    backend=backend,
                    shots_path=shots,
                    top_n=int(top_n_val),
                    min_reviews=int(min_rev),
                    max_new_tokens=int(max_tok),
                    temperature=float(temp)
                )
                
                # Parse recommendations
                r1, r2, r3 = parse_recommendations(result)
                
                # Format stats
                stats_text = f"""
                **Generation Statistics:**
                - Time: {stats.get('time', 0):.1f}s ({stats.get('time', 0)/60:.1f} minutes)
                - Tokens: ~{int(stats.get('tokens', 0))}
                - Speed: {stats.get('tokens_per_sec', 0):.1f} tokens/sec
                - Model: {model}
                """
                
                return r1, r2, r3, result, stats_text
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                if "Ollama" in str(e):
                    error_msg += "\n\n**To fix:** Run `ollama serve &` in terminal"
                return "", "", "", error_msg, ""
        
        generate_btn.click(
            fn=generate_wrapper,
            inputs=[model_choice, cluster_dropdown, top_n, min_reviews, max_tokens, temperature, shots_path],
            outputs=[rec1, rec2, rec3, output_md, stats_md]
        )
        
        # Examples
        gr.Examples(
            examples=[
                ["Ollama Qwen 2.5 7B (Best)", "E-Readers (Kindle)", 3, 50, 1500, 0.7, "prompts/examples_v3.json"],
                ["Flan-T5-large (Fast)", "Fire Tablets (7-inch)", 3, 100, 1500, 0.7, "prompts/examples_v3.json"],
                ["Ollama Qwen 2.5 7B (Best)", "Fire Tablets (HD 8-inch)", 3, 50, 1500, 0.7, "prompts/examples_v3.json"],
            ],
            inputs=[model_choice, cluster_dropdown, top_n, min_reviews, max_tokens, temperature, shots_path],
        )
    
    return demo


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 Starting RoboReviews Unified App")
    print("="*80)
    
    print("\nBuilding interface...")
    demo = build_unified_app()
    
    print("\n🌐 Launching web interface...")
    print("="*80 + "\n")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,  # Single port for all models
        share=False
    )
