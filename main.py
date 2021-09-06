import torch
import warnings
warnings.filterwarnings('ignore')

from preprocesses.preprocess2 import x_train, y_train, x_test, y_test, n_features
from models.model2 import IsVacinated
from utils import getMetrics

use_saved_model = True
model_name = "prep2_model2_1"


model = IsVacinated(n_features)
if use_saved_model:
    model.load_state_dict(torch.load(f"./saved/{model_name}.pth"))
    model.eval()
    y_pred = model.predict(x_test)
    _, y_true = model.tensorify(y=y_test)
    y_pred = model.predict(x_test)
    getMetrics(y_true, y_pred, model_name)
    print(f"accuracy = {model.accuracy(x_test, y_test):.4f}")
else:
    '''
    lr = 0.001
    epochs = 500
    criterion = nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    '''
    model.fit(x_train, y_train, x_test, y_test, epochs=270, lr=0.001)
    y_pred = model.predict(x_test)
    _, y_true = model.tensorify(y=y_test)
    y_pred = model.predict(x_test)
    getMetrics(y_true, y_pred, model_name)
    print(f"accuracy = {model.accuracy(x_test, y_test):.4f}")
    torch.save(model.state_dict(), f"./saved/{model_name}.pth")