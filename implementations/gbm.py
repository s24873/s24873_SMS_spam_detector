import pandas as pd
import numpy as np
import h2o
from h2o.estimators import H2OGradientBoostingEstimator
from sklearn.metrics import accuracy_score, mean_absolute_error

h2o.init()

train_path = '../ml_data/train.csv'
test_path = '../ml_data/test.csv'
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
train_h2o = h2o.H2OFrame(train_df)
test_h2o = h2o.H2OFrame(test_df)
train_h2o['label'] = train_h2o['label'].asfactor()
test_h2o['label'] = test_h2o['label'].asfactor()

target = 'label'
features = [col for col in train_df.columns if col != target]

gbm_model = H2OGradientBoostingEstimator()
gbm_model.train(x=features, y=target, training_frame=train_h2o)

preds = gbm_model.predict(test_h2o).as_data_frame()

# obliczanie metryk jakosci modelu
pred_classes = preds['predict'].apply(lambda x: 1 if x == '1' else 0).values

# prawdziwe etykiety
y_true = test_df[target].values

accuracy = accuracy_score(y_true, pred_classes)
mae = mean_absolute_error(y_true, pred_classes)

print(f"Dokładność: {accuracy:.4f}")
print(f"Średni błąd bezwzględny (MAE): {mae:.4f}")

# dodatkowe metryki
try:
    auc = gbm_model.auc()  # AUC
    print(f"AUC: {auc:.4f}")
except KeyError:
    print("AUC nie jest dostępne dla tego modelu")

logloss = gbm_model.logloss()
gini = gbm_model.gini()

print(f"LogLoss: {logloss:.4f}")
print(f"Gini: {gini:.4f}")


