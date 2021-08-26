import numpy as np
import torch
import torch.nn as nn


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
        criterion = criterion()
        optimizer = optimizer(self.parameters(), lr=lr)
        x_train, y_train = self.tensorify(x_train, y_train)
        for epoch in range(1, epochs+1):
            y_pred = self(x_train)
            loss = criterion(y_pred, y_train)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if epoch % 10 == 0:
                acc = self.accuracy(x_test, y_test)
                print(f"epoch = {epoch}, loss = {loss.item():.4f}, acc = {acc:.4f}")
    
    def tensorify(self, x, y):
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))
        y = y.view(y.shape[0], 1)
        
        return x, y
    
    def accuracy(self, x_test, y_test):
        x_test, y_test = self.tensorify(x_test, y_test)
        with torch.no_grad():
            y_pred = self(x_test)
            y_pred_cls = y_pred.round()
            accuracy = y_pred_cls.eq(y_test).sum() / float(y_test.shape[0])
            return accuracy