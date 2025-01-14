from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import pandas as pd
import h2o
from airflow.utils.dates import days_ago
from h2o.automl import H2OAutoML
from h2o.estimators import H2OGradientBoostingEstimator
from oauth2client.service_account import ServiceAccountCredentials
import gspread

CREDENTIALS_PATH = "/opt/airflow/credentials.json"
h2o.init()

def load_data_from_google_sheets():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_PATH, scope)
    client = gspread.authorize(creds)
    sheet_names = {'spam_train': 'processed_data/train.csv', 'spam_test': 'processed_data/test.csv'}
    os.makedirs("processed_data", exist_ok=True)

    for sheet_name in sheet_names:
        try:
            sheet = client.open(sheet_name).sheet1
            data = sheet.get_all_values()
            if data:
                df = pd.DataFrame(data[1:], columns=data[0])
                print(f"Loaded data from '{sheet_name}', shape: {df.shape}.")
            else:
                print(f"No data found in sheet '{sheet_name}'")
        except gspread.exceptions.SpreadsheetNotFound:
            print(f"Sheet '{sheet_name}' not found")
        except Exception as e:
            print(f"Error: '{sheet_name}': {e}")


def train_h2o_model():
    train_path = 'processed_data/train.csv'
    test_path = 'processed_data/test.csv'

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print("Train data columns:", train_df.columns)
    print("Test data columns:", test_df.columns)

    train_h2o = h2o.H2OFrame(train_df)
    test_h2o = h2o.H2OFrame(test_df)

    if 'v1' not in train_h2o.columns:
        print("Column v1 not found!")
        return

    train_h2o['v1'] = train_h2o['v1'].asfactor()
    test_h2o['v1'] = test_h2o['v1'].asfactor()

    x = train_h2o.columns[:-1]
    y = 'v1'
    aml = H2OAutoML(max_models=7, seed=42)
    aml.train(x=x, y=y, training_frame=train_h2o)

    perf = aml.leader.model_performance(test_h2o)
    accuracy = perf.accuracy()[0][1]
    print("Accuracy: ",accuracy)
    model_path = h2o.save_model(model=aml.leader, path="models", force=True)
    print(f"Model saved into: {model_path}")

    os.makedirs("reports", exist_ok=True)
    report_path = "reports/evaluation_report.txt"
    with open(report_path, "w") as report_file:
        report_file.write(f"Accuracy: {accuracy}\n")
        report_file.write(str(perf))
    print(f"Report saved into: {report_path}")


default_args = {
    'owner': 'piotr',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1),
}

with DAG('model_training_dag', default_args=default_args, schedule_interval=None) as dag:
    load_data = PythonOperator(
        task_id='load_data_from_google_sheets',
        python_callable=load_data_from_google_sheets
    )

    train_model = PythonOperator(
        task_id='train_h2o_model',
        python_callable=train_h2o_model
    )

    load_data >> train_model
