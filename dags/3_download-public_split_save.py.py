import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import gspread
from oauth2client.service_account import ServiceAccountCredentials


CREDENTIALS_PATH = "/opt/airflow/credentials.json"

def split_data():
    file_path = "/opt/airflow/spam.csv"
    if os.path.exists(file_path):
        print(f"Plik {file_path} istnieje!")
    else:
        print(f"Plik {file_path} nie istnieje!")

    data = pd.read_csv(file_path)

    test_size = 0.3
    test_index = int(len(data) * (1 - test_size))

    train = data.iloc[:test_index]
    test = data.iloc[test_index:]

    train.to_csv("/tmp/spam_train.csv", index=False)
    test.to_csv("/tmp/spam_test.csv", index=False)


def upload_to_google_sheets():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_PATH, scope)
    client = gspread.authorize(creds)

    train_sheet = client.open("spam_train").sheet1
    test_sheet = client.open("spam_test").sheet1
    train_data = pd.read_csv("/tmp/spam_train.csv")
    test_data = pd.read_csv("/tmp/spam_test.csv")

    train_sheet.update([train_data.columns.values.tolist()] + train_data.values.tolist())
    test_sheet.update([test_data.columns.values.tolist()] + test_data.values.tolist())


default_args = {
    "owner": "piotr",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "split_data_dag",
        default_args=default_args,
        description="DAG do podziału danych i zapisu do Google Sheets",
        schedule_interval=None,
        start_date=days_ago(1),
        tags=["spam-detector"],
) as dag:
    split_data_task = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
    )

    upload_to_sheets_task = PythonOperator(
        task_id="upload_to_google_sheets",
        python_callable=upload_to_google_sheets,
    )

    split_data_task >> upload_to_sheets_task

