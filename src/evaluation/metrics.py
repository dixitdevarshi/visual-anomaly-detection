import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import List, Dict


def compute_auroc(labels: List[int], scores: List[float]) -> float:
    """
    Compute image-level AUROC.
    Args:
        labels: ground truth binary labels (0=normal, 1=anomaly)
        scores: anomaly scores from PatchCore
    Returns:
        AUROC score between 0 and 1
    """
    return roc_auc_score(labels, scores)


def compute_average_precision(labels: List[int], scores: List[float]) -> float:
    """
    Compute average precision score.
    Args:
        labels: ground truth binary labels
        scores: anomaly scores
    Returns:
        AP score between 0 and 1
    """
    return average_precision_score(labels, scores)


def compute_category_metrics(
    labels: List[int],
    scores: List[float],
    category: str
) -> Dict:
    """
    Compute all metrics for a single MVTec category.
    Args:
        labels: ground truth binary labels
        scores: anomaly scores
        category: MVTec category name
    Returns:
        dict with category metrics
    """
    auroc = compute_auroc(labels, scores)
    ap = compute_average_precision(labels, scores)

    result = {
        "category": category,
        "auroc": round(auroc, 4),
        "average_precision": round(ap, 4),
        "n_normal": labels.count(0),
        "n_anomaly": labels.count(1),
        "n_total": len(labels)
    }

    print(f"[{category}] AUROC: {auroc:.4f} | AP: {ap:.4f} | "
          f"Normal: {labels.count(0)} | Anomaly: {labels.count(1)}")

    return result


def compute_overall_metrics(category_results: List[Dict]) -> Dict:
    """
    Compute mean metrics across all categories.
    Args:
        category_results: list of dicts from compute_category_metrics
    Returns:
        dict with mean and std of each metric
    """
    aurocs = [r["auroc"] for r in category_results]
    aps = [r["average_precision"] for r in category_results]

    overall = {
        "mean_auroc": round(np.mean(aurocs), 4),
        "std_auroc": round(np.std(aurocs), 4),
        "mean_average_precision": round(np.mean(aps), 4),
        "std_average_precision": round(np.std(aps), 4),
        "n_categories": len(category_results)
    }

    print(f"\nOverall AUROC: {overall['mean_auroc']:.4f} ± {overall['std_auroc']:.4f}")
    print(f"Overall AP:    {overall['mean_average_precision']:.4f} ± {overall['std_average_precision']:.4f}")

    return overall


def save_results(
    category_results: List[Dict],
    overall: Dict,
    save_path: str
):
    """
    Save results to JSON file.
    Args:
        category_results: per category metrics
        overall: overall metrics
        save_path: path to save JSON
    """
    import json
    from pathlib import Path

    output = {
        "overall": overall,
        "per_category": category_results
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Results saved to {save_path}")