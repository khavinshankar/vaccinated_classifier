import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def pre_na(df):
    cols = list(df.columns)

    for col in cols:
        if df[col].dtype == int or df[col].dtype == float:
            df[col].fillna(value = df[col].mean(), inplace = True)
        else:
            df[col].fillna(value = 'X', inplace = True)

        df[col] = encoder.fit_transform(df[col])

    return df

def engineer(df):
    cols = list(df.columns)
    behavioral = []
    for col in cols:
        if 'behavioral' in col:
            behavioral.append(col)
    
    df['behavioral'] = df[behavioral].sum(axis = 1)
    df['doctor'] = df['doctor_recc_h1n1'] + df['doctor_recc_seasonal'] 
    df['mind'] = df['health_worker'] + df['h1n1_knowledge'] + df['h1n1_concern']

    return df

x = pd.read_csv("./data/train.csv")
y = pd.read_csv("./data/train_labels.csv")

encoder = LabelEncoder()

x_mod = pre_na(x)
x_final = engineer(x_mod)

x_final = np.array(x_final)
y_final = np.array(y["vacc_h1n1_f"])

n_samples, n_features = x_final.shape
x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, random_state=48, test_size=0.2)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)