import h2o
from h2o.automl import H2OAutoML
import pandas as pd

h2o.init()

train_data = pd.read_csv("../ml_data/train.csv")
test_data = pd.read_csv("../ml_data/test.csv")
train_data_h2o = h2o.H2OFrame(train_data)
test_data_h2o = h2o.H2OFrame(test_data)

train_data_h2o['label'] = train_data_h2o['label'].asfactor()
test_data_h2o['label'] = test_data_h2o['label'].asfactor()

x = train_data_h2o.columns[:-1]
y = 'label'

aml = H2OAutoML(max_models=7, seed=42)
aml.train(x=x, y=y, training_frame=train_data_h2o)
# ranking modeli
lb = aml.leaderboard
print(lb)
# ocena modelu na zbiorze test
predictions = aml.leader.predict(test_data_h2o)
perf = aml.leader.model_performance(test_data_h2o)

print(perf)
accuracy = perf.accuracy()[0][1]
print(f"Dokładność modelu: {accuracy}")

