import numpy as np
import torch
import torch.nn as nn


class IsVacinated(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 2),
            nn.Softmax()
        )
        
    def forward(self, x):
        return self.network(x)
    
    def fit(self, x_train, y_train, x_test, y_test, epochs=75, lr=0.01, criterion=nn.CrossEntropyLoss, optimizer=torch.optim.Adam):
        criterion = criterion()
        optimizer = optimizer(self.parameters(), lr=lr)
        x_train, y_train = self.tensorify(x_train, y_train)
        for epoch in range(1, epochs+1):
            y_pred = self(x_train)
            loss = criterion(y_pred, y_train.type(torch.LongTensor))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if epoch % 10 == 0:
                acc = self.accuracy(x_test, y_test)
                print(f"epoch = {epoch}, loss = {loss.item():.4f}, acc = {acc:.4f}")
    
    def predict(self, x_test):
        x_test, _ = self.tensorify(x_test)
        with torch.no_grad():
            y_pred = self(x_test)
        
        return y_pred[:,1]
    
    def tensorify(self, x=np.array([]), y=np.array([])):
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))
        y = y.view(y.shape[0])
        
        return x, y
    
    def accuracy(self, x_test, y_test):
        x_test, y_test = self.tensorify(x_test, y_test)
        with torch.no_grad():
            y_pred = self(x_test)
            values, indicies = torch.max(y_pred, dim=1)
            accuracy = torch.tensor(torch.sum(indicies == y_test).item() / len(y_pred))
            return accuracy

