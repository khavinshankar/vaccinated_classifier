import torch
import warnings
warnings.filterwarnings('ignore')

from preprocesses.preprocess1 import x_train, y_train, x_test, y_test, n_features
from models.model2 import IsVacinated

use_saved_model = False
model_name = "prep1_model2_1"


model = IsVacinated(n_features)
if use_saved_model:
    model.load_state_dict(torch.load(f"./saved/{model_name}.pth"))
    model.eval()
    print(f"accuracy = {model.accuracy(x_test, y_test)}")
else:
    '''
    lr = 0.001
    epochs = 500
    criterion = nn.CrossEntropyLoss
    optimizer = torch.optim.Adam
    '''
    model.fit(x_train, y_train, x_test, y_test)
    print(f"accuracy = {model.accuracy(x_test, y_test)}")
    torch.save(model.state_dict(), f"./saved/{model_name}.pth")