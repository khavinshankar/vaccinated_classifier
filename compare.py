from matplotlib import pyplot as plt
import numpy as np
from numpy.core.fromnumeric import size
from preprocesses.preprocess1 import (
    x_train as x_train1,
    y_train as y_train1,
    x_test as x_test1,
    y_test as y_test1,
    n_features as n_features1
)
from preprocesses.preprocess2 import (
    x_train as x_train2,
    y_train as y_train2,
    x_test as x_test2,
    y_test as y_test2,
    n_features as n_features2
)
from models.model1 import IsVacinated as Model1
from models.model2 import IsVacinated as Model2
from random_forest import y_test as y_test_rf, y_pred as y_pred_rf
from sklearn.metrics import classification_report, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

model1_prep1 = Model1(n_features1)
model1_prep2 = Model1(n_features2)
model2_prep1 = Model2(n_features1)
model2_prep2 = Model2(n_features2)

model1_prep1_metrics = model1_prep1.fit(x_train1, y_train1, x_test1,
                                        y_test1, epochs=600, lr=0.001)
model1_prep2_metrics = model1_prep2.fit(x_train2, y_train2, x_test2,
                                        y_test2, epochs=600, lr=0.001)
model2_prep1_metrics = model2_prep1.fit(x_train1, y_train1, x_test1,
                                        y_test1, epochs=600, lr=0.001)
model2_prep2_metrics = model2_prep2.fit(x_train2, y_train2, x_test2,
                                        y_test2, epochs=600, lr=0.001)
random_forest_metrics = classification_report(
    y_test_rf, y_pred_rf.round(), output_dict=True)

plt.plot(model1_prep1_metrics["accuracy_scores"], label="Model1 Prep1")
plt.plot(model1_prep2_metrics["accuracy_scores"], label="Model1 Prep2")
plt.plot(model2_prep1_metrics["accuracy_scores"], label="Model2 Prep1")
plt.plot(model2_prep2_metrics["accuracy_scores"], label="Model2 Prep2")
plt.legend(loc=4)
plt.title("Accuracy")
plt.savefig("./metrics/accuracy-line.png")
plt.show()

plt.plot(model1_prep1_metrics["precision_scores"], label="Model1 Prep1")
plt.plot(model1_prep2_metrics["precision_scores"], label="Model1 Prep2")
plt.plot(model2_prep1_metrics["precision_scores"], label="Model2 Prep1")
plt.plot(model2_prep2_metrics["precision_scores"], label="Model2 Prep2")
plt.legend(loc=4)
plt.title("Precision")
plt.savefig("./metrics/precision-line.png")
plt.show()

plt.plot(model1_prep1_metrics["recall_scores"], label="Model1 Prep1")
plt.plot(model1_prep2_metrics["recall_scores"], label="Model1 Prep2")
plt.plot(model2_prep1_metrics["recall_scores"], label="Model2 Prep1")
plt.plot(model2_prep2_metrics["recall_scores"], label="Model2 Prep2")
plt.legend(loc=4)
plt.title("Recall")
plt.savefig("./metrics/recall-line.png")
plt.show()

plt.plot(model1_prep1_metrics["f1_scores"], label="Model1 Prep1")
plt.plot(model1_prep2_metrics["f1_scores"], label="Model1 Prep2")
plt.plot(model2_prep1_metrics["f1_scores"], label="Model2 Prep1")
plt.plot(model2_prep2_metrics["f1_scores"], label="Model2 Prep2")
plt.legend(loc=4)
plt.title("F1 Score")
plt.savefig("./metrics/f1_score-line.png")
plt.show()

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(model1_prep1_metrics["accuracy_scores"], label="Model1 Prep1")
plt.plot(model1_prep2_metrics["accuracy_scores"], label="Model1 Prep2")
plt.plot(model2_prep1_metrics["accuracy_scores"], label="Model2 Prep1")
plt.plot(model2_prep2_metrics["accuracy_scores"], label="Model2 Prep2")
plt.title("Accuracy", size=10)

plt.subplot(2, 2, 2)
plt.plot(model1_prep1_metrics["precision_scores"], label="Model1 Prep1")
plt.plot(model1_prep2_metrics["precision_scores"], label="Model1 Prep2")
plt.plot(model2_prep1_metrics["precision_scores"], label="Model2 Prep1")
plt.plot(model2_prep2_metrics["precision_scores"], label="Model2 Prep2")
plt.title("Precision", size=10)

plt.subplot(2, 2, 3)
plt.plot(model1_prep1_metrics["recall_scores"], label="Model1 Prep1")
plt.plot(model1_prep2_metrics["recall_scores"], label="Model1 Prep2")
plt.plot(model2_prep1_metrics["recall_scores"], label="Model2 Prep1")
plt.plot(model2_prep2_metrics["recall_scores"], label="Model2 Prep2")
plt.title("Recall", size=10)

plt.subplot(2, 2, 4)
plt.plot(model1_prep1_metrics["f1_scores"], label="Model1 Prep1")
plt.plot(model1_prep2_metrics["f1_scores"], label="Model1 Prep2")
plt.plot(model2_prep1_metrics["f1_scores"], label="Model2 Prep1")
plt.plot(model2_prep2_metrics["f1_scores"], label="Model2 Prep2")
plt.title("F1 Score", size=10)

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.86,
                    wspace=0.4,
                    hspace=0.4)
plt.figlegend(["Model1 Prep1", "Model1 Prep2", "Model2 Prep1", "Model2 Prep2"],
              loc='upper center', ncol=4, labelspacing=0.5)
plt.savefig("./metrics/classification_report-line.png")
plt.show()

# bar graph
barWidth = 0.1
fig = plt.subplots(figsize=(12, 8))

M1P1 = [model1_prep1_metrics["accuracy_scores"][-1], model1_prep1_metrics["precision_scores"]
        [-1], model1_prep1_metrics["recall_scores"][-1], model1_prep1_metrics["f1_scores"][-1]]
M1P2 = [model1_prep2_metrics["accuracy_scores"][-1], model1_prep2_metrics["precision_scores"]
        [-1], model1_prep2_metrics["recall_scores"][-1], model1_prep2_metrics["f1_scores"][-1]]
M2P1 = [model2_prep1_metrics["accuracy_scores"][-1], model2_prep1_metrics["precision_scores"]
        [-1], model2_prep1_metrics["recall_scores"][-1], model2_prep1_metrics["f1_scores"][-1]]
M2P2 = [model2_prep2_metrics["accuracy_scores"][-1], model2_prep2_metrics["precision_scores"]
        [-1], model2_prep2_metrics["recall_scores"][-1], model2_prep2_metrics["f1_scores"][-1]]
RF = [random_forest_metrics["accuracy"], random_forest_metrics['weighted avg']['precision'],
      random_forest_metrics['weighted avg']['recall'], random_forest_metrics['weighted avg']['f1-score']]

br1 = np.arange(len(M1P1))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]

plt.bar(br1, M1P1, width=barWidth, label='Model1 Prep1')
plt.bar(br2, M1P2, width=barWidth, label='Model1 Prep2')
plt.bar(br3, M2P1, width=barWidth, label='Model2 Prep1')
plt.bar(br4, M2P2, width=barWidth, label='Model2 Prep2')
plt.bar(br5, RF, width=barWidth, label='Random Forest')


plt.xticks([r + barWidth for r in range(len(M1P1))],
           ['Accuracy', 'Precision', 'Recall', 'F1 Score'])

plt.legend()
plt.savefig("./metrics/classification_report-bar.png")
plt.show()


# auc graph
roc_m1p1 = roc_curve(y_test1, model1_prep1.predict(x_test1))
roc_m1p2 = roc_curve(y_test2, model1_prep2.predict(x_test2))
roc_m2p1 = roc_curve(y_test1, model2_prep1.predict(x_test1))
roc_m2p2 = roc_curve(y_test2, model2_prep2.predict(x_test2))
roc_rf = roc_curve(y_test_rf, y_pred_rf)

plt.plot(roc_m1p1[0], roc_m1p1[1],
         label=f"Model1 Prep1 = {auc(roc_m1p1[0], roc_m1p1[1]):.3f}")
plt.plot(roc_m1p2[0], roc_m1p2[1],
         label=f"Model1 Prep2 = {auc(roc_m1p2[0], roc_m1p2[1]):.3f}")
plt.plot(roc_m2p1[0], roc_m2p1[1],
         label=f"Model2 Prep1 = {auc(roc_m2p1[0], roc_m2p1[1]):.3f}")
plt.plot(roc_m2p2[0], roc_m2p2[1],
         label=f"Model2 Prep2 = {auc(roc_m2p2[0], roc_m2p2[1]):.3f}")
plt.plot(roc_rf[0], roc_rf[1],
         label=f"Random Forest = {auc(roc_rf[0], roc_rf[1]):.3f}")
plt.legend(loc=4)
plt.title("AUC")
plt.savefig("./metrics/auc.png")
plt.show()
