import os
import re
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from hybrid_nn import HybridEvaluator, HybridVideoDataset, hybrid_collate_fn

THUMBNAIL_URL_MEDIUM = "thumbnail_url_medium"
TITLE = "title"
DURATION = "duration"
TITLE_LENGTH = "title_length"
VID_HOUR_OF_DAY = "vid_hour_of_day"
VID_DAY_OF_WEEK = "vid_day_of_week"
SUBSCRIBER_COUNT = "subscriber_count"
CHANNEL_VIEWS = "channel_views"
VIDEO_COUNT = "video_count"
CHANNEL_AGE_YEARS = "channel_age_years"
VIEWS = "views"

def sine_squared_lr(epoch):
    warmup_epochs = 10
    if epoch < warmup_epochs:
        return float(epoch) / float(max(1, warmup_epochs))
    else:
        T_cur = epoch - warmup_epochs
        T_total = 100 - warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * T_cur / T_total))

df = pd.read_csv("video_data_cleaned.csv", encoding="utf-8")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

batch_dir = "thumbnail_batches"
batch_files = [
    f for f in os.listdir(batch_dir)
    if f.startswith("batch_") and f.endswith(".pt")
]
batch_files.sort(
    key=lambda x: int(re.search(r"batch_(\d+)\.pt", x).group(1))
)
batches = []
for fname in batch_files:
    path = os.path.join(batch_dir, fname)
    tensor = torch.load(path, weights_only=True)
    batches.append(tensor)
thumbnail_tensors_stack = torch.cat(batches, dim=0)
print(f"Loaded {thumbnail_tensors_stack.size(0)} thumbnails.")
numeric_feats = [DURATION, TITLE_LENGTH, SUBSCRIBER_COUNT, CHANNEL_VIEWS, VIDEO_COUNT, CHANNEL_AGE_YEARS]

dataset = HybridVideoDataset(
    df=df,
    thumbnail_tensors=thumbnail_tensors_stack,
    numeric_features=numeric_feats,
    title_col=TITLE,
    hour_col=VID_HOUR_OF_DAY,
    dow_col=VID_DAY_OF_WEEK,
    label_col=VIEWS,
)

K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

all_indices = list(range(len(dataset)))
trainval_idx, test_idx = train_test_split(
    all_indices,
    test_size=0.05,
    random_state=42,
    shuffle=True
)

trainval_ds = torch.utils.data.Subset(dataset, trainval_idx)
test_ds = torch.utils.data.Subset(dataset, test_idx)

num_epochs = 100
warmup_steps = int(0.005 * num_epochs)
decay_steps = num_epochs - warmup_steps
lr_period = num_epochs // 4
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(trainval_idx)):
    print(f"Starting fold {fold+1}/{K}...")
    fold_train_idx = [trainval_idx[i] for i in train_idx]
    fold_val_idx   = [trainval_idx[i] for i in val_idx]
    patience = 0
    prev_loss = 100.0

    train_ds = torch.utils.data.Subset(dataset, fold_train_idx)
    val_ds   = torch.utils.data.Subset(dataset, fold_val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        collate_fn=hybrid_collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=128,
        shuffle=False,
        collate_fn=hybrid_collate_fn,
        pin_memory=True
    )

    model = HybridEvaluator(num_numeric_features=len(numeric_feats), num_classes=8, device=device).to(device)

    labels = torch.tensor(train_ds.dataset.df.loc[train_ds.indices, VIEWS].to_numpy(), dtype=torch.long)
    class_counts = torch.bincount(labels)
    num_classes = len(class_counts)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * num_classes
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=1e-4
    )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=sine_squared_lr
    )

    for epoch in range(num_epochs):
        model.train()

        total_train_loss = 0.0
        total_val_loss = 0.0
        exit = 0
        train_correct = 0
        train_total = 0
        val_correct = 0
        val_total = 0
        val_all_labels = []
        val_all_preds = []

        for batch in train_loader:
            images = batch["images"].to(device)
            titles = batch["titles"]
            numeric = batch["numeric"].to(device)
            hour = batch["hour"].to(device)
            dow = batch["dow"].to(device)
            train_labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                images,
                titles,
                numeric,
                hour,
                dow
            )

            train_loss = criterion(outputs, train_labels)
            train_loss.backward()
            optimizer.step()

            train_preds = outputs.argmax(dim=1)
            total_train_loss += train_loss.item()
            train_correct += (train_preds == train_labels).sum().item()
            train_total += train_labels.size(0)
        
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            for batch in val_loader:

                images = batch["images"].to(device)
                titles = batch["titles"]
                numeric = batch["numeric"].to(device)
                hour = batch["hour"].to(device)
                dow = batch["dow"].to(device)
                val_labels = batch["labels"].to(device)

                val_outputs = model(
                    images,
                    titles,
                    numeric,
                    hour,
                    dow
                )
                val_loss = criterion(val_outputs, val_labels)

                val_preds = val_outputs.argmax(dim=1)
                val_correct += (val_preds == val_labels).sum().item()
                val_total += val_labels.size(0)
                total_val_loss += val_loss.item()

                val_all_labels.append(val_labels.cpu())
                val_all_preds.append(val_preds.cpu())

        val_all_labels = torch.cat(val_all_labels)
        val_all_preds  = torch.cat(val_all_preds)
        cm = confusion_matrix(val_all_labels.numpy(), val_all_preds.numpy())

        avg_train_loss = total_train_loss / train_total
        avg_val_loss = total_val_loss / val_total
        avg_train_acc = train_correct / train_total
        avg_val_acc = val_correct / val_total
        
        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Training Loss: {avg_train_loss:.4f} "
            f"Training Acc: {avg_train_acc:.4f} "
            f"Validation Loss: {avg_val_loss:.4f} "
            f"Validation Acc: {avg_val_acc:.4f} "
        )

        if avg_val_loss < prev_loss:
            prev_loss = avg_val_loss
            patience = 0
        else:
            patience += 1
            print("No improvement, increasing patience to", patience)
            if patience >= 3:
                patience = 0
                torch.save(model.state_dict(), f"youtube_video_evaluator_fold_{fold}.pth")
                fold_results.append({
                    "fold": fold,
                    "val_loss": avg_val_loss,
                    "val_acc": avg_val_acc,
                    "val_all_labels": val_all_labels,
                    "val_all_preds": val_all_preds
                })
                print("Early stopping triggered at epoch", epoch+1, "with validation accuracy", avg_val_acc, "\nconfusion matrix:\n", cm)
                break

all_labels = [
    label
    for fold in fold_results
    for label in fold["val_all_labels"]
]

all_preds = [
    pred
    for fold in fold_results
    for pred in fold["val_all_preds"]
]

print(classification_report(all_labels, all_preds))
print(confusion_matrix(all_labels, all_preds))

# TESTING
best_fold = min(fold_results, key=lambda x: x["val_loss"])
best_model_path = f"youtube_video_evaluator_fold_{best_fold['fold']}.pth"
print(f"\nLoading best model from fold {best_fold['fold']} with val loss {best_fold['val_loss']:.4f}")
model.load_state_dict(torch.load(best_model_path))
test_loader = DataLoader(
    test_ds,
    batch_size=128,
    shuffle=False,
    collate_fn=hybrid_collate_fn,
    pin_memory=True
)
test_criterion = torch.nn.CrossEntropyLoss()

test_loss = 0.0
test_all_labels = []
test_all_preds = []
test_correct = 0
test_total = 0

model.eval()
with torch.no_grad():
    for batch in test_loader:
        images = batch["images"].to(device)
        titles = batch["titles"]
        numeric = batch["numeric"].to(device)
        hour = batch["hour"].to(device)
        dow = batch["dow"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            images,
            titles,
            numeric,
            hour,
            dow
        )

        loss = test_criterion(outputs, labels)
        test_loss += loss.item()
        preds = outputs.argmax(dim=1)
        
        test_all_labels.append(labels.cpu())
        test_all_preds.append(preds.cpu())

        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

avg_test_loss = test_loss / test_total
avg_test_acc = test_correct / test_total
test_all_labels = torch.cat(test_all_labels).numpy()
test_all_preds = torch.cat(test_all_preds).numpy()

print(
    f"Testing Loss: {avg_test_loss:.4f} "
    f"Testing Acc: {avg_test_acc:.4f}"
)
print(confusion_matrix(test_all_labels, test_all_preds))
print(classification_report(test_all_labels, test_all_preds))