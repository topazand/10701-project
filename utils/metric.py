import numpy as np
import matplotlib.pyplot as plt
# calculate accuracy, precision, recall, F1 score and confusion matrix
def calculate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    confusion_matrix = np.array([[tn, fp], [fn, tp]])

    return accuracy, precision, recall, f1_score, confusion_matrix

def print_metrics(y_true, y_pred):
    accuracy, precision, recall, f1_score, confusion_matrix = calculate_metrics(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix)

def plot_ROC(y_true, y_pred):
    thresholds = np.linspace(0, 1, 1000)
    tpr = [1]
    fpr = [1]
    for thresh in thresholds:
        y_pred_thresh = (y_pred >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred_thresh == 1))
        tn = np.sum((y_true == 0) & (y_pred_thresh == 0))
        fp = np.sum((y_true == 0) & (y_pred_thresh == 1))
        fn = np.sum((y_true == 1) & (y_pred_thresh == 0))
        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
    tpr.append(0)
    fpr.append(0)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()