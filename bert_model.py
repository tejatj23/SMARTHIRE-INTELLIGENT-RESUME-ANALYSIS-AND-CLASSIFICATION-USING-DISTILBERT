import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    get_linear_schedule_with_warmup,
)

MODEL_WEIGHTS_PATH = "bert_model.pth"
LABEL_MAP_PATH     = "bert_label_map.json"

class ResumeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }

def build_label_map(data):
    labels = sorted(data["label"].unique().tolist())
    label_map   = {label: i for i, label in enumerate(labels)}
    reverse_map = {i: label for label, i in label_map.items()}
    return label_map, reverse_map


def train(data, tokenizer, label_map, batch_size=8, epochs=4):
    texts = data["text"].tolist()
    y     = [label_map[l] for l in data["label"]]

    dataset    = ResumeDataset(texts, y, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label_map),
    )
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

    total_steps   = len(dataloader) * epochs
    warmup_steps  = max(1, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg = total_loss / len(dataloader)
        print(f"  Epoch {epoch + 1}/{epochs}  avg loss: {avg:.4f}")

    return model

def save(model, label_map):
    torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f)
    print("Model and label map saved.")


def load_model_and_maps(tokenizer):

    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)

    reverse_map = {int(i): label for label, i in label_map.items()}

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label_map),
    )
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location="cpu"))
    print("Saved model loaded.")
    return model, label_map, reverse_map

data      = pd.read_csv("training_data.csv")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

if os.path.exists(MODEL_WEIGHTS_PATH) and os.path.exists(LABEL_MAP_PATH):
    model, label_map, reverse_map = load_model_and_maps(tokenizer)
else:
    print("Training BERT model...")
    label_map, reverse_map = build_label_map(data)
    model = train(data, tokenizer, label_map, batch_size=8, epochs=4)
    save(model, label_map)

def predict_domain(text: str) -> str:
    model.eval()
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return reverse_map[pred]