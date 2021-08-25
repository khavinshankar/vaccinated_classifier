import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

x = pd.read_csv("./data/train.csv")
y = pd.read_csv("./data/train_labels.csv")

x_mod = x.replace(np.nan, 0.0)
encodings = {
    "opinion_h1n1_vacc_effective": {
        "Not At All Effective": -2,
        "Not Very Effective": -1,
        "Refused": 0,
        "0.0": 0, 
        "Dont Know": 0,
        "Somewhat Effective": 1,
        "Very Effective": 2,
    },
    "opinion_h1n1_risk": {
        "Very Low": -2,
        "Somewhat Low": -1,
        "Refused": 0,
        "0.0": 0,
        "Dont Know": 0,
        "Somewhat High": 1,
        "Very High": 2,
    },
    "opinion_h1n1_sick_from_vacc": {
        "Not At All Worried": -2,
        "Not Very Worried": -1,
        "Refused": 0,
        "0.0": 0,
        "Dont Know": 0,
        "Somewhat Worried": 1,
        "Very Worried": 2,
    },
    "opinion_seas_vacc_effective": {
        "Not At All Effective": -2,
        "Not Very Effective": -1,
        "Refused": 0,
        "0.0": 0, 
        "Dont Know": 0,
        "Somewhat Effective": 1,
        "Very Effective": 2,
    },
    "opinion_seas_risk": {
        "Very Low": -2,
        "Somewhat Low": -1,
        "Refused": 0,
        "0.0": 0,
        "Dont Know": 0,
        "Somewhat High": 1,
        "Very High": 2,
    },
    "opinion_seas_sick_from_vacc": {
        "Not At All Worried": -2,
        "Not Very Worried": -1,
        "Refused": 0,
        "0.0": 0,
        "Dont Know": 0,
        "Somewhat Worried": 1,
        "Very Worried": 2,
    },
    "agegrp": { # try one-hot encoding if this doesn't work
        "6 Months - 9 Years": 1,
        "10 - 17 Years": 2,
        "18 - 34 Years": 3,
        "35 - 44 Years": 4,
        "45 - 54 Years": 5,
        "55 - 64 Years": 6,
        "65+ Years": 7,
    },
    "employment_status": {
        "Unemployed": -1,
        "0.0": 0,
        "Not in Labor Force": 1,
        "Employed": 2,
    },
    "census_msa": {
        "Non-MSA": 1,
        "MSA, Not Principle City": 2,
        "MSA, Principle City": 3,
    },
}

x_mod = x_mod.replace(encodings)
x_final = pd.get_dummies(x_mod, columns=["employment_industry", "employment_occupation", "state"], prefix=["emp_ind", "emp_occ", "state"])
y_final = np.array(y["vacc_h1n1_f"])

n_samples, n_features = x_final.shape

x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, random_state=48, test_size=0.2)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

saved_model = True
PATH = "./model1.pth"

class IsVacinated(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.linear1 = nn.Linear(input_features, 64)
        self.linear2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.linear1(x))
        out = self.linear2(out)
        return torch.sigmoid(out)
    
model = IsVacinated(n_features)

def test(model):
    with torch.no_grad():
        y_pred = model(x_test)
        y_pred_cls = y_pred.round()

        accuracy = y_pred_cls.eq(y_test).sum() / float(y_test.shape[0])
        print(f"accuracy = {accuracy:.4f}")

if not saved_model:
    lr = 0.001
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = 300
    for epoch in range(1, epochs+1):
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 10 == 0: print(f"epoch = {epoch}, loss = {loss.item():.4f}")

    torch.save(model.state_dict(), PATH)
    test(model)
else:
    model.load_state_dict(torch.load(PATH))
    model.eval()
    test(model)