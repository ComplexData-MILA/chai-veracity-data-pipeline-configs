"""Fine-tune a small transformer for 3-way text classification."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm.auto import tqdm


class JSONLDataset(Dataset):
    """Loads a JSONL file of {"text": ..., "is_feasible": ...} records."""

    def __init__(self, path: str, tokenizer, max_length: int):
        self.samples: list[dict] = []
        with open(path) as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(sample["is_feasible"], dtype=torch.long),
        }


def evaluate(model, dataloader, loss_fn, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(dataloader), correct / total


def train_one_epoch(model, dataloader, optimizer, scheduler, loss_fn, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=3, pad_token_id=tokenizer.pad_token_id
    ).to(device)

    train_ds = JSONLDataset(
        os.path.join(args.data_dir, "train.jsonl"), tokenizer, args.max_length
    )
    val_ds = JSONLDataset(
        os.path.join(args.data_dir, "val.jsonl"), tokenizer, args.max_length
    )
    test_ds = JSONLDataset(
        os.path.join(args.data_dir, "test.jsonl"), tokenizer, args.max_length
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_ckpt_path = Path(args.output_dir) / "best_model.pt"
    best_ckpt_path_hf = Path(args.output_dir) / "best_model"
    best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(best_ckpt_path_hf)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device
        )
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        print(
            f"epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt_path)
            model.save_pretrained(best_ckpt_path_hf)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

    model.load_state_dict(torch.load(best_ckpt_path, weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
    print(f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a classifier")
    parser.add_argument(
        "--data_dir", type=str, default=os.path.expandvars("$SCRATCH/classifier_data")
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.expandvars("$SCRATCH/classifier_models/default"),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
