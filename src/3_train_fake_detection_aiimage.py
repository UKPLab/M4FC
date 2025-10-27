import os
import json
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import timm
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from utils import dataset_loader
from torch.utils.data import Subset


# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_labels(dataset):
    neg_count = sum(1 for item in dataset if int(item["is_manipulated_fake"] or item["is_ai_generated"]) == 1)
    pos_count = len(dataset) - neg_count
    return pos_count, neg_count


class FakeImageDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]["image_path"]
        label = int(not (self.data[idx]["is_manipulated_fake"] or self.data[idx]["is_ai_generated"]))
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)
            return image_tensor, label
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a default black image as fallback
            dummy_image = torch.zeros((3, 224, 224))
            return dummy_image, label


def balanced_subset(dataset, max_per_class=None, seed=42):
    """
    Create a balanced subset with equal positive and negative samples.
    """
    labels = [dataset[i][1] for i in range(len(dataset))]
    pos_indices = [i for i, l in enumerate(labels) if l == 1]
    neg_indices = [i for i, l in enumerate(labels) if l == 0]

    min_len = min(len(pos_indices), len(neg_indices))
    if max_per_class:
        min_len = min(min_len, max_per_class)

    random.seed(seed)
    pos_sampled = random.sample(pos_indices, min_len)
    neg_sampled = random.sample(neg_indices, min_len)

    balanced_indices = pos_sampled + neg_sampled
    random.shuffle(balanced_indices)

    return Subset(dataset, balanced_indices)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.criterion(inputs, targets)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean()


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            # Move inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    metrics = {
        'loss': running_loss / len(dataloader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, pos_label=1, zero_division=0),
        'recall': recall_score(all_labels, all_preds, pos_label=1, zero_division=0),
        'f1': f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    }

    return metrics


def plot_metrics(train_metrics, val_metrics, metric_name, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics, label=f'Train {metric_name}')
    plt.plot(val_metrics, label=f'Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'Training and Validation {metric_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/{metric_name}_plot.png")
    plt.close()


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories
    model_name = "dafilab_ai_image_detector"  # Simplified name for directories
    os.makedirs(f"{args.output_dir}/{model_name}", exist_ok=True)
    os.makedirs(f"{args.output_dir}/{model_name}/plots", exist_ok=True)

    # Load pretrained model from timm
    print(f"Loading model: {args.model_name_or_path}")
    model = timm.create_model(args.model_name_or_path, pretrained=True, num_classes=2)

    # Check if model has an adequate number of classes for binary classification
    num_classes = model.get_classifier().out_features
    if num_classes != 2:
        print(f"Adjusting model output from {num_classes} to 2 classes")
        # Get the classifier name (might vary by model architecture)
        if hasattr(model, 'head'):
            classifier_name = 'head'
        elif hasattr(model, 'fc'):
            classifier_name = 'fc'
        elif hasattr(model, 'classifier'):
            classifier_name = 'classifier'
        else:
            raise ValueError("Could not identify the classifier head of the model")

        # Get in_features of current classifier
        in_features = getattr(model, classifier_name).in_features

        # Replace the classifier with a new one for binary classification
        setattr(model, classifier_name, nn.Linear(in_features, 2))

    model.to(device)

    # Get default transform configuration for the model
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=True)

    # Load dataset
    train_set_raw, dev_set_raw, test_set_raw = dataset_loader(args.file, args.task)
    print(f"Dataset sizes: Train={len(train_set_raw)}, Dev={len(dev_set_raw)}, Test={len(test_set_raw)}")

    # Count labels distribution
    train_pos, train_neg = count_labels(train_set_raw)
    dev_pos, dev_neg = count_labels(dev_set_raw)
    test_pos, test_neg = count_labels(test_set_raw)

    print(
        f"Train set: Total={len(train_set_raw)}, Positive={train_pos}, Negative={train_neg}, Ratio={train_pos / len(train_set_raw):.2f}")
    print(
        f"Dev set: Total={len(dev_set_raw)}, Positive={dev_pos}, Negative={dev_neg}, Ratio={dev_pos / len(dev_set_raw):.2f}")
    print(
        f"Test set: Total={len(test_set_raw)}, Positive={test_pos}, Negative={test_neg}, Ratio={test_pos / len(test_set_raw):.2f}")

    # Create datasets with appropriate transforms
    train_dataset = FakeImageDataset(train_set_raw, transform)

    # For validation and test, use a transform without augmentation
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = FakeImageDataset(dev_set_raw, eval_transform)
    test_dataset = FakeImageDataset(test_set_raw, eval_transform)

    balanced_test_dataset = balanced_subset(test_dataset, max_per_class=500)

    # Create weighted sampler for handling class imbalance
    if args.weighted_sampling and train_pos > 0 and train_neg > 0:
        train_labels = [int(not (item["is_manipulated_fake"] or item["is_ai_generated"])) for item in train_set_raw]
        class_counts = [train_neg, train_pos]
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = class_weights[torch.tensor(train_labels)]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
        print("Using weighted sampling to handle class imbalance")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Set up loss function
    if args.focal_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2)
        print("Using Focal Loss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard Cross Entropy Loss")

    # Set up optimizer with different parameter groups
    # Get classifier name based on model architecture
    if hasattr(model, 'head'):
        classifier_name = 'head'
    elif hasattr(model, 'fc'):
        classifier_name = 'fc'
    elif hasattr(model, 'classifier'):
        classifier_name = 'classifier'
    else:
        raise ValueError("Could not identify the classifier head of the model")

    # Differential learning rates - train classification head faster than backbone
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in getattr(model, classifier_name).named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.classifier_lr,
        },
        {
            "params": [p for n, p in getattr(model, classifier_name).named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.classifier_lr,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if classifier_name not in n and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if classifier_name not in n and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.lr,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters)

    # Use OneCycleLR scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[args.classifier_lr, args.classifier_lr, args.lr, args.lr],
        steps_per_epoch=len(train_loader),
        epochs=args.num_train_epochs,
        pct_start=0.1  # Warm up for the first 10% of training
    )

    # Initialize tracking variables
    best_val_metric = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Initial evaluation
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(
        f"📊 Initial test metrics: Accuracy={test_metrics['accuracy']:.4f}, Precision={test_metrics['precision']:.4f}, "
        f"Recall={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}")

    # Training loop
    for epoch in range(args.num_train_epochs):
        print(f"\n✏️ Epoch {epoch + 1}/{args.num_train_epochs}")
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []

        train_loop = tqdm(train_loader, desc="Training")

        for inputs, labels in train_loop:
            # Move inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            # Calculate training accuracy
            _, preds = torch.max(outputs, 1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

            # Update progress bar
            train_loop.set_postfix(loss=loss.item())

        # Calculate training metrics for this epoch
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        print(f"📊 Train loss: {avg_train_loss:.4f}, Train accuracy: {train_accuracy:.4f}")

        # Validation
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_metrics['loss'])
        val_accuracies.append(val_metrics['accuracy'])

        print(f"📈 Validation metrics: Loss={val_metrics['loss']:.4f}, Accuracy={val_metrics['accuracy']:.4f}, "
              f"F1={val_metrics['f1']:.4f}, Precision={val_metrics['precision']:.4f}, Recall={val_metrics['recall']:.4f}")

        # Early stopping based on precision
        current_metric = val_metrics['precision']
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            patience_counter = 0
            # Save the best model
            best_model_path = os.path.join(args.output_dir, model_name, "best_model.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'epoch': epoch,
            }, best_model_path)
            print("✅ Best model updated and saved!")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                # If learning rate reduction is enabled
                if args.reduce_lr_on_plateau:
                    # Reduce learning rate by factor of 5
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = max(param_group['lr'] / 5, 1e-6)

                    print(f"⚠️ Reducing learning rate")

                    # Reset patience counter with fewer attempts
                    patience_counter = args.patience // 2
                else:
                    print(f"⚠️ Early stopping! No improvement for {args.patience} epochs")
                    break

        # Save checkpoint for this epoch
        if epoch % args.save_every == 0 or epoch == args.num_train_epochs - 1:
            checkpoint_path = os.path.join(args.output_dir, model_name, f"checkpoint-{epoch + 1}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
            }, checkpoint_path)

    # Plot training curves
    plot_metrics(train_losses, val_losses, 'Loss', f"{args.output_dir}/{model_name}/plots")
    plot_metrics(train_accuracies, val_accuracies, 'Accuracy', f"{args.output_dir}/{model_name}/plots")

    # Load best model for final testing
    best_model_path = os.path.join(args.output_dir, model_name, "best_model.pth")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final testing
    test_metrics = evaluate(model, test_loader, criterion, device)

    print("\n🏁 Training completed!")
    print(f"📊 Test metrics: Accuracy={test_metrics['accuracy']:.4f}, Precision={test_metrics['precision']:.4f}, "
          f"Recall={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}")

    # Save final results
    results = {
        "model": args.model_name_or_path,
        "best_val_accuracy": float(best_val_metric),
        "best_epoch": len(train_losses) - patience_counter,
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "training_history": {
            "train_losses": [float(x) for x in train_losses],
            "val_losses": [float(x) for x in val_losses],
            "train_accuracies": [float(x) for x in train_accuracies],
            "val_accuracies": [float(x) for x in val_accuracies]
        },
        "hyperparameters": vars(args)
    }

    with open(os.path.join(args.output_dir, model_name, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data/M4FC.json", help="File path")
    parser.add_argument("--task", type=str, default="fake_detection", help="Task name")
    parser.add_argument("--model_name_or_path", type=str, default="hf_hub:Dafilab/ai-image-detector",
                        help="Path to pretrained model or model identifier from timm model zoo")
    parser.add_argument("--output_dir", type=str, default="./finetuned_models",
                        help="The output directory where the fine-tuned model will be saved")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for model backbone")
    parser.add_argument("--classifier_lr", type=float, default=5e-5, help="Learning rate for classifier head")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for gradient clipping")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--focal_loss", action="store_true", help="Use focal loss for training")
    parser.add_argument("--weighted_sampling", action="store_true", help="Use weighted sampling for class imbalance")
    parser.add_argument("--reduce_lr_on_plateau", action="store_true", help="Reduce LR when validation metric plateaus")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")

    args = parser.parse_args()

    main(args)
