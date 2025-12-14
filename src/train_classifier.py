# src/train_classifier.py
# DistilBERT sentiment classifier with MPS-friendly memory settings

from __future__ import annotations
import os, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
    get_linear_schedule_with_warmup,
)

# --------------------------- Config --------------------------- #
INP_PARQUET = Path("artifacts/clean_reviews.parquet")
INP_CSV     = Path("artifacts/clean_reviews.csv")  # fallback
OUT_DIR     = Path("artifacts/clf")
MODEL_NAME  = "distilbert-base-uncased"
SEED        = 42
TEST_SIZE   = 0.2

# Memory-safe knobs
MAX_LEN     = 256          # cut sequence length to reduce attention memory
BATCH_TRAIN = 8            # smaller batch
BATCH_EVAL  = 16
GRAD_ACC    = 2            # keeps effective train batch ~= 16
EPOCHS      = 3
LR          = 5e-5
WARMUP_RATIO = 0.06

LABELS = ["negative", "neutral", "positive"]
L2I = {l: i for i, l in enumerate(LABELS)}
I2L = {i: l for l, i in L2I.items()}

# Optional: lower MPS watermark if not set
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0")

# --------------------------- Data --------------------------- #
def load_clean_frame() -> pd.DataFrame:
    if INP_PARQUET.exists():
        df = pd.read_parquet(INP_PARQUET)
    elif INP_CSV.exists():
        df = pd.read_csv(INP_CSV)
    else:
        raise SystemExit(
            f"Dataset not found. Expected {INP_PARQUET} or {INP_CSV}. "
            "Run: python src/preprocess.py --input 'data/*.csv' --out artifacts"
        )

    if "text" not in df.columns:       raise SystemExit("Expected 'text' column.")
    if "sentiment" not in df.columns:  raise SystemExit("Expected 'sentiment' column.")

    df = df[df["text"].notna() & (df["text"].astype(str).str.strip() != "")]
    df = df[df["sentiment"].isin(LABELS)].copy()
    if len(df) == 0: raise SystemExit("No usable rows after filtering.")

    df["label"] = df["sentiment"].map(L2I).astype(int)
    return df[["text", "label"]].reset_index(drop=True)

def build_hf_datasets(df: pd.DataFrame):
    train_df, val_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df["label"]
    )
    ds_train = Dataset.from_pandas(train_df, preserve_index=False)
    ds_val   = Dataset.from_pandas(val_df,   preserve_index=False)
    return ds_train, ds_val

# --------------------------- Tokenization --------------------------- #
def tokenize_fn_builder(tokenizer):
    def _tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)
    return _tokenize

# --------------------------- Metrics --------------------------- #
def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision_macro": p, "recall_macro": r, "f1_macro": f1}

# --------------------------- Optimizer (compat) --------------------------- #
from torch.optim import AdamW as TorchAdamW

class AdamWCompat(TorchAdamW):
    def train(self):  # no-op for accelerate compatibility
        return self
    def eval(self):   # no-op
        return self

def build_optimizer_and_scheduler(model, train_dataset_len: int, args: TrainingArguments):
    steps_per_epoch = max(1, int(np.ceil(train_dataset_len / args.per_device_train_batch_size)))
    total_steps = steps_per_epoch * int(args.num_train_epochs)
    warmup_steps = max(1, int(WARMUP_RATIO * total_steps))

    optimizer = AdamWCompat(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    return optimizer, scheduler

# --------------------------- Main --------------------------- #
def main():
    warnings.filterwarnings("ignore")
    set_seed(SEED)

    # Clear MPS cache after previous crashes
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    # 1) Data
    df = load_clean_frame()
    print(f"Rows: {len(df):,}")
    ds_train, ds_val = build_hf_datasets(df)

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenize_fn = tokenize_fn_builder(tokenizer)
    ds_train = ds_train.map(tokenize_fn, batched=True, remove_columns=["text"])
    ds_val   = ds_val.map(tokenize_fn,   batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3) Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=I2L,
        label2id=L2I,
    )
    # Save memory
    model.gradient_checkpointing_enable()

    # 4) Args
    args = TrainingArguments(
        output_dir=str(OUT_DIR / "runs"),
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=SEED,
        optim="adamw_torch",   # explicit
        fp16=False,
        bf16=False,
    )

    # 5) Optimizer + scheduler
    optimizer, scheduler = build_optimizer_and_scheduler(model, len(ds_train), args)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
    )

    # 6) Train
    train_out = trainer.train()
    print("Finished training.")

    # 7) Evaluate
    eval_out = trainer.evaluate()
    print("Eval:", eval_out)

    # 8) Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    (OUT_DIR / "metrics.json").write_text(json.dumps({"train": train_out.metrics, "eval": eval_out}, indent=2))
    (OUT_DIR / "label_map.json").write_text(json.dumps({"id2label": I2L, "label2id": L2I}, indent=2))
    print(f"Saved model and artifacts → {OUT_DIR}")

if __name__ == "__main__":
    main()
