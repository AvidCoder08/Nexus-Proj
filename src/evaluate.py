"""
Evaluation utilities: compute metrics and generate reports/plots.
"""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def compute_metrics(y_true, y_probs, threshold=0.5):
    """Compute common metrics for binary classification.

    Args:
        y_true (array-like): ground truth labels (0/1)
        y_probs (array-like): predicted probabilities for positive class
        threshold (float): decision threshold

    Returns:
        dict with accuracy, precision, recall, f1, auc
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    y_pred = (y_probs >= threshold).astype(int)

    report = classification_report(y_true, y_pred, output_dict=True)
    auc = roc_auc_score(y_true, y_probs)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        'classification_report': report,
        'roc_auc': float(auc),
        'confusion_matrix': cm.tolist()
    }
    return metrics


if __name__ == '__main__':
    # small sanity test
    y_true = [0,1,1,0,1,0]
    y_probs = [0.1,0.9,0.7,0.4,0.6,0.2]
    print(compute_metrics(y_true, y_probs))
