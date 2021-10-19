from category_encoders import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

from preprocesses.preprocess2 import x_train, y_train, x_test, y_test
from utils import getMetrics

pipe_ord = make_pipeline(
    OrdinalEncoder(), 
    SimpleImputer(), 
    RandomForestClassifier(
        n_estimators=150, 
        random_state=10, 
        max_depth=15, 
        oob_score=True, 
        n_jobs=-1, 
        criterion="gini", 
        min_samples_split=5, 
        max_features=6
    )
)

pipe_ord.fit(x_train, y_train)
y_pred = pipe_ord.predict_proba(x_test)[:, 1]

getMetrics(y_test, y_pred, "random-forest_1")
print(f"accuracy = {pipe_ord.score(x_test, y_test):.4f}")
