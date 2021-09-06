import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def getMetrics(y_true, y_pred, file_name=""):
    print(y_true, y_pred)
    report = classification_report(y_true, y_pred.round())
    matrix = confusion_matrix(y_true, y_pred.round())
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    auc_ = auc(fpr, tpr)

    plt.figure(figsize=(5, 5), dpi=100)
    plt.plot(fpr, tpr, linestyle='-', label=f'AUC = {auc_:.3f}')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    if file_name:
        file = open(f'metrics/{file_name}-report.txt', 'w+')
        file.write(f'Classification Report: \n {report} \n \nConfusion Matrix: \n {matrix}')
        file.close()
        plt.savefig(f'metrics/{file_name}-auc.png')

    print(report)
    print(matrix)
    plt.show()