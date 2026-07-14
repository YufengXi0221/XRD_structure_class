import argparse
import csv
import math
from pathlib import Path

import torch
from torch import nn



MODEL_DIR = r"C:\Users\xyfxy\Desktop\XRD data\XRD_old\1.2m_soup_nopool_230"
DATASET_DIR = r"C:\Users\xyfxy\Desktop\XRD data\RRUFF\RRUFF"
#DATASET_DIR = r"C:\Users\xyfxy\Desktop\XRD data\RRUFF\RRUFF"
OUTPUT_CSV = ""
LABELS_CSV = "labels230.csv"
BATCH_SIZE = 256
TOP_K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class NoPoolCNN(nn.Module):
    def __init__(self, input_shape=(1,)):
        super().__init__()
        in_channels = input_shape if isinstance(input_shape, int) else input_shape[0]
        self.CNN = nn.Sequential(
            nn.Conv1d(in_channels, 80, 100, 5),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(80, 80, 50, 5),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(80, 80, 25, 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, obs):
        return self.CNN(obs)


class Predictor(nn.Module):
    def __init__(self, input_shape=(12160,), output_shape=(7,)):
        super().__init__()
        input_dim = input_shape if isinstance(input_shape, int) else math.prod(input_shape)
        output_dim = output_shape if isinstance(output_shape, int) else math.prod(output_shape)
        self.MLP = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 2300),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2300, 1150),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1150, output_dim),
        )

    def forward(self, obs):
        return self.MLP(obs)


def load_model(model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    if "1.MLP.7.weight" not in state_dict:
        raise ValueError("This checkpoint does not match the expected NoPoolCNN + Predictor model.")

    num_classes = state_dict["1.MLP.7.weight"].shape[0]
    model = nn.Sequential(NoPoolCNN(), Predictor(input_shape=12160, output_shape=num_classes))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, num_classes


def resolve_model_path(model_input):
    model_path = Path(model_input)
    if model_path.is_dir():
        model_path = model_path / "model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return model_path


def resolve_features_path(features_input=None, dataset_dir=None):
    if features_input:
        features_path = Path(features_input)
    else:
        features_path = Path(dataset_dir) / "features.csv"

    if features_path.is_dir():
        features_path = features_path / "features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {features_path}")
    return features_path


def resolve_labels_path(labels_input, dataset_dir=None):
    if not labels_input:
        return None

    labels_path = Path(labels_input)
    if labels_path.exists():
        return labels_path

    if dataset_dir and not labels_path.is_absolute():
        labels_path = Path(dataset_dir) / labels_input
        if labels_path.exists():
            return labels_path

    print(f"Label CSV not found: {labels_path}. Prediction will continue without labels.")
    return None


def parse_feature_row(row):
    return [float(x) for x in row if x.strip() != ""]


def parse_label_row(row):
    values = [float(x) for x in row if x.strip() != ""]
    if len(values) == 1:
        return int(values[0])
    return max(range(len(values)), key=values.__getitem__)


def load_labels(labels_csv):
    labels = []
    with open(labels_csv, "r", newline="") as fin:
        reader = csv.reader(fin)
        for row in reader:
            if not row:
                continue
            try:
                labels.append(parse_label_row(row))
            except ValueError:
                if not labels:
                    continue
                raise
    return labels


def predict_csv(model, features_csv, output_csv, labels_csv, device, batch_size=256, top_k=5):
    top_k = max(1, int(top_k))
    sample_index = 0
    batch = []
    batch_indices = []
    labels = load_labels(labels_csv) if labels_csv else None
    correct = 0
    total = 0

    with open(features_csv, "r", newline="") as fin, open(output_csv, "w", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = ["sample_index", "predicted_class", "confidence"]
        if labels is not None:
            header.append("true_class")
        for rank in range(1, top_k + 1):
            header.extend([f"top{rank}_class", f"top{rank}_probability"])
        writer.writerow(header)

        def flush_batch():
            nonlocal batch, batch_indices, correct, total
            if not batch:
                return

            x = torch.tensor(batch, dtype=torch.float32, device=device).unsqueeze(1)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)
                k = min(top_k, probs.shape[1])
                top_probs, top_classes = torch.topk(probs, k=k, dim=1)

            for local_i, original_i in enumerate(batch_indices):
                pred_class = int(top_classes[local_i, 0].item())
                confidence = float(top_probs[local_i, 0].item())
                row = [original_i, pred_class, confidence]
                if labels is not None:
                    true_class = labels[original_i]
                    row.append(true_class)
                    total += 1
                    if pred_class == true_class:
                        correct += 1
                for cls, prob in zip(top_classes[local_i].tolist(), top_probs[local_i].tolist()):
                    row.extend([int(cls), float(prob)])
                writer.writerow(row)

            batch = []
            batch_indices = []

        for row in reader:
            if not row:
                continue
            try:
                features = parse_feature_row(row)
            except ValueError:

                if sample_index == 0:
                    continue
                raise

            batch.append(features)
            batch_indices.append(sample_index)
            sample_index += 1

            if len(batch) >= batch_size:
                flush_batch()

        flush_batch()
    return correct, total


def main():
    parser = argparse.ArgumentParser(description="Run XRD NoPoolCNN model prediction on a feature CSV.")
    parser.add_argument("--model", default=MODEL_DIR, help="Path to model.pth, or a model folder containing model.pth.")
    parser.add_argument("--dataset-dir", default=DATASET_DIR, help="Dataset folder containing features.csv.")
    parser.add_argument("--features", help="CSV file containing one XRD feature vector per row.")
    parser.add_argument("--labels", default=LABELS_CSV, help="Optional label CSV path. If blank or missing, prediction runs without labels.")
    parser.add_argument("--output", default=OUTPUT_CSV, help="Output CSV path. If blank, predictions.csv is written in the dataset folder.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    features_path = resolve_features_path(args.features, args.dataset_dir)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = features_path.with_name("predictions.csv")

    device = torch.device(args.device)
    model, num_classes = load_model(model_path, device)
    print(f"Loaded {num_classes}-class model on {device}.")
    labels_path = resolve_labels_path(args.labels, args.dataset_dir)
    if labels_path:
        print(f"Using labels from {labels_path}")
    else:
        print("No labels file found. Accuracy will not be calculated.")

    correct, total = predict_csv(
        model=model,
        features_csv=features_path,
        output_csv=output_path,
        labels_csv=labels_path,
        device=device,
        batch_size=args.batch_size,
        top_k=args.top_k,
    )
    print(f"Wrote predictions to {output_path}")
    if total:
        print(f"Accuracy: {correct / total:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
