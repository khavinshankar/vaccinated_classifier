import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report


class IsVacinated(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

    def fit(self, x_train, y_train, x_test, y_test, epochs=60, lr=0.01, criterion=nn.BCELoss, optimizer=torch.optim.Adam):
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        criterion = criterion()
        optimizer = optimizer(self.parameters(), lr=lr)

        x_train, y_train = self.tensorify(x_train, y_train)
        x_test, y_test = self.tensorify(x_test, y_test)

        for epoch in range(1, epochs+1):
            y_pred = self(x_train)
            loss = criterion(y_pred, y_train)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if epoch % 10 == 0:
                accuracy, precision, recall, f1 = self.metrics(x_test, y_test)
                accuracy_scores.append(accuracy)
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)

        return {
            "accuracy_scores": accuracy_scores,
            "precision_scores": precision_scores,
            "recall_scores": recall_scores,
            "f1_scores": f1_scores
        }

    def predict(self, x_test):
        x_test, _ = self.tensorify(x_test)
        with torch.no_grad():
            y_pred = self(x_test)

        return y_pred

    def tensorify(self, x=np.array([]), y=np.array([])):
        x = x if torch.is_tensor(x) else torch.from_numpy(x.astype(np.float32))
        y = y if torch.is_tensor(y) else torch.from_numpy(y.astype(np.float32))
        y = y.view(y.shape[0], 1)

        return x, y

    def metrics(self, x_test, y_test):
        x_test, y_test = self.tensorify(x_test, y_test)
        with torch.no_grad():
            y_pred = self.predict(x_test)
            report = classification_report(
                y_test, y_pred.round(), output_dict=True)

        return report['accuracy'], report['weighted avg']['precision'], report['weighted avg']['recall'], report['weighted avg']['f1-score']
