import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
        "Not in Labor Force": 0,
        "Employed": 1,
    },
    "census_msa": {
        "Non-MSA": 1,
        "MSA, Not Principle City": 2,
        "MSA, Principle City": 3,
    },
}

x_mod = x_mod.replace(encodings)
x_mod = x_mod.drop(["state", "census_msa"], axis=1)
x_final = pd.get_dummies(x_mod, columns=["employment_industry", "employment_occupation"], prefix=["emp_ind", "emp_occ"])
y_final = np.array(y["vacc_h1n1_f"])

n_samples, n_features = x_final.shape

x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, random_state=48, test_size=0.2)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
