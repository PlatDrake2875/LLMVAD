import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
)

from main import EvalModes


def _create_metadata_yaml(
    cache_path: Path,
    raw_predictions: list[dict[str, float]],
    expected: list[dict[str, int]],
    eval_mode: EvalModes,
    threshold: float | None = None,
) -> None:
    """
    Create a YAML sidecar file with metadata about the cached results.
    """
    labels = ["B1", "B2", "B4", "B5", "B6", "G", "A"]

    total_samples = len(expected)

    label_frequencies = {}
    for label in labels:
        count = sum(exp[label] for exp in expected)
        label_frequencies[label] = {
            "count": count,
            "percentage": round(count / total_samples * 100, 2)
            if total_samples > 0
            else 0,
        }

    metadata = {
        "created_at": datetime.now().isoformat(),
        "eval_mode": eval_mode.value,
        "total_samples": total_samples,
        "ground_truth_distribution": label_frequencies,
    }

    if threshold is not None:
        metadata["threshold"] = threshold

        binary_predictions = {}
        for label in labels:
            pred_count = sum(
                1 for pred in raw_predictions if pred.get(label, 0) > threshold
            )
            binary_predictions[label] = {
                "predicted_positive": pred_count,
                "percentage": round(pred_count / total_samples * 100, 2)
                if total_samples > 0
                else 0,
            }
        metadata["binary_predictions"] = binary_predictions

    yaml_path = cache_path.with_suffix(".yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    print(f"Metadata saved to {yaml_path}")


def get_raw_predictions(
    cache_path: Path | None = None,
    eval_mode: EvalModes = EvalModes.VIDEO_SIMPLE,
    use_cache: bool = True,
) -> tuple[list[dict[str, float]], list[dict[str, int]]]:
    if cache_path is None:
        cache_path = Path("cache") / eval_mode.value

    raw_cache_path = Path("cache") / "results" / f"{eval_mode.value}_raw.joblib"
    if use_cache and raw_cache_path.exists():
        print(f"Loading cached raw predictions from {raw_cache_path}")
        return joblib.load(raw_cache_path)

    labels = ["B1", "B2", "B4", "B5", "B6", "G", "A"]

    raw_predictions = []
    expected = []

    for cache_file in sorted(cache_path.glob("*.joblib")):
        data = joblib.load(cache_file)
        response_text = data["response_text"].replace("\n", " ")
        pred_data = json.loads(response_text)
        video_name = data["video_name"]

        pred_scores = {}
        for item in pred_data:
            if "tag_id" in item and "score" in item:
                pred_scores[item["tag_id"]] = float(item["score"])

        for label in labels:
            if label not in pred_scores:
                pred_scores[label] = 0.0

        raw_predictions.append(pred_scores)

        exp_labels = video_name.split("label_")[1].split(".mp4")[0].split("-")
        exp = {label: 1 if label in exp_labels else 0 for label in labels}
        expected.append(exp)

    if use_cache:
        raw_cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump((raw_predictions, expected), raw_cache_path)
        print(f"Raw predictions cached to {raw_cache_path}")

        _create_metadata_yaml(
            cache_path=raw_cache_path,
            raw_predictions=raw_predictions,
            expected=expected,
            eval_mode=eval_mode,
        )

    return raw_predictions, expected


def get_accuracy_report(
    cache_path: Path | None = None,
    eval_mode: EvalModes = EvalModes.VIDEO_SIMPLE,
    threshold: float = 0.7,
    use_cache: bool = True,
) -> dict[str, dict[str, float]]:
    if cache_path is None:
        cache_path = Path("cache") / eval_mode.value

    results_cache_path = (
        Path("cache") / "results" / f"{eval_mode.value}_threshold_{threshold}.joblib"
    )
    if use_cache and results_cache_path.exists():
        print(f"Loading cached results from {results_cache_path}")
        return joblib.load(results_cache_path)

    raw_predictions, expected = get_raw_predictions(
        cache_path=cache_path, eval_mode=eval_mode, use_cache=use_cache
    )

    labels = ["B1", "B2", "B4", "B5", "B6", "G", "A"]

    predictions = []
    for raw_pred in raw_predictions:
        pred = {
            label: 1 if raw_pred.get(label, 0) > threshold else 0 for label in labels
        }
        predictions.append(pred)

    results = {}
    for label in labels:
        tp = tn = fp = fn = 0
        for pred, exp in zip(predictions, expected):
            if pred[label] == 1 and exp[label] == 1:
                tp += 1
            elif pred[label] == 0 and exp[label] == 0:
                tn += 1
            elif pred[label] == 1 and exp[label] == 0:
                fp += 1
            else:
                fn += 1

        total = tp + tn + fp + fn
        results[label] = {
            "accuracy": (tp + tn) / total if total > 0 else 0.0,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        }

    if use_cache:
        results_cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(results, results_cache_path)
        print(f"Results cached to {results_cache_path}")

        # Create metadata YAML sidecar
        _create_metadata_yaml(
            cache_path=results_cache_path,
            raw_predictions=raw_predictions,
            expected=expected,
            eval_mode=eval_mode,
            threshold=threshold,
        )

    return results


def print_accuracy_report(
    results=None, eval_mode: EvalModes = EvalModes.VIDEO_SIMPLE, use_cache: bool = True
) -> None:
    if results is None:
        results = get_accuracy_report(eval_mode=eval_mode, use_cache=use_cache)

    print("Class\tAccuracy\tPrecision\tRecall")
    print("-" * 40)

    for label in ["B1", "B2", "B4", "B5", "B6", "G", "A"]:
        r = results[label]
        print(
            f"{label}\t{r['accuracy']:.3f}\t\t{r['precision']:.3f}\t\t{r['recall']:.3f}"
        )

    overall = sum(r["accuracy"] for r in results.values()) / len(results)
    print(f"\nOverall Accuracy: {overall:.3f}")


def compute_pr_curves(
    cache_path: Path | None = None,
    save_plot: bool = False,
    eval_mode: EvalModes = EvalModes.VIDEO_SIMPLE,
    use_cache: bool = True,
) -> dict[str, float]:
    raw_predictions, expected = get_raw_predictions(
        cache_path, eval_mode=eval_mode, use_cache=use_cache
    )
    labels = ["B1", "B2", "B4", "B5", "B6", "G", "A"]

    ap_scores = {}
    plt.figure(figsize=(10, 8))

    print(f"\nData distribution (total samples: {len(expected)}):")
    for label in labels:
        y_true = [exp[label] for exp in expected]
        positive_count = sum(y_true)
        total_count = len(y_true)
        print(
            f"{label}: {positive_count}/{total_count} positive samples ({positive_count / total_count:.1%})"
        )

    print("\nSample prediction scores:")
    for label in labels:
        scores = []
        for pred in raw_predictions[:5]:
            scores.append(pred.get(label, 0.0))
        print(f"{label}: {scores}")

    for label in labels:
        y_true = [exp[label] for exp in expected]

        y_scores = []
        for pred in raw_predictions:
            score = pred.get(label, 0.0)
            y_scores.append(score)

        if sum(y_true) > 0:
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)
            ap_scores[label] = ap

            pr_auc = auc(recall, precision)

            plt.plot(
                recall, precision, label=f"{label} (AP = {ap:.3f}, AUC = {pr_auc:.3f})"
            )
        else:
            ap_scores[label] = 0.0

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curves ({eval_mode.value})")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    if save_plot:
        # Save to cache directory with eval_mode in filename
        plot_path = Path("cache") / "results" / f"{eval_mode.value}_pr_curves.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"PR curve saved to {plot_path}")
    else:
        plt.show()

    return ap_scores


if __name__ == "__main__":
    eval_mode = EvalModes.ONTOLOGICAL_CATEGORIES

    print_accuracy_report(eval_mode=eval_mode)

    ap_scores = compute_pr_curves(save_plot=True, eval_mode=eval_mode)

    print("\nAverage Precision (AP) Scores:")
    print("-" * 40)
    for label, score in ap_scores.items():
        print(f"{label}: {score:.3f}")
    mean_ap = sum(ap_scores.values()) / len(ap_scores)
    print(f"\nMean Average Precision (mAP): {mean_ap:.3f}")
