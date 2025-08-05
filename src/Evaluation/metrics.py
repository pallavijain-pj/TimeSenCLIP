import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import numpy as np 

def class_wise_accuracy(logits_stk, labels_stk, class_names):
    # Convert logits to predicted class labels
    predicted_labels = torch.argmax(logits_stk, dim=1)
    
    # Calculate accuracy scores
    micro_accuracy = accuracy_score(labels_stk.cpu().numpy(), predicted_labels.cpu().numpy())
    
    # Calculate precision, recall, F1-score, and support for each class
    precision, recall, f1, support = precision_recall_fscore_support(labels_stk.cpu().numpy(), predicted_labels.cpu().numpy(), labels=range(len(class_names)))
    
    # Calculate macro and weighted average metrics
    macro_accuracy = sum(precision) / len(precision)
    weighted_accuracy = sum(precision[i] * support[i] for i in range(len(precision))) / sum(support)
    class_accuracies = []
    total_accuracy = 0.0
    for class_idx in range(len(class_names)):
        # Select the samples belonging to the current class
        class_mask = (labels_stk.cpu().numpy() == class_idx)
        class_correct = (predicted_labels.cpu().numpy()[class_mask] == class_idx).sum()
        class_accuracy = class_correct / support[class_idx] if support[class_idx] > 0 else 0.0
        total_accuracy += class_accuracy
        class_accuracies.append(class_accuracy)
    
    avg_cls_acc = total_accuracy/len(class_names)
    # Create a dictionary to store results for each class
    class_results = {}
    for i in range(len(class_names)):
        class_results[class_names[i]] = {
            "Accuracy": class_accuracies[i],
            "Precision": precision[i],
            "Recall": recall[i],
            "F1-Score": f1[i]
        }
    return class_results, avg_cls_acc

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
