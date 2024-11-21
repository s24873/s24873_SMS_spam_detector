from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import timedelta
from airflow.utils.dates import days_ago
from googleapiclient.errors import HttpError

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
CREDENTIALS_PATH = '/opt/airflow/credentials.json'


def get_gsheet_service():
    creds = Credentials.from_service_account_file(CREDENTIALS_PATH, scopes=SCOPES)
    service = build('sheets', 'v4', credentials=creds)
    return service


def load_data_from_google_sheets(**kwargs):
    sheet_id_1 = '1fq820VFHfqjMllet10Z7wXdoOrvHhOLi8d5ru8mJWG4'
    sheet_id_2 = '1s03JjEbI5y1VdMNNHK5ratHV9urTkppiRMvJVVKFFiA'
    sheets = ['spam_train', 'spam_test']

    dfs = {}
    service = get_gsheet_service()

    for sheet in sheets:
        if sheet == 'spam_train':
            range_ = 'spam_train!A1:B3901'
        elif sheet == 'spam_test':
            range_ = 'spam_test!A1:B1673'

        try:
            result = service.spreadsheets().values().get(
                spreadsheetId=sheet_id_1 if sheet == 'spam_train' else sheet_id_2, range=range_).execute()
            values = result.get('values', [])

            if not values:
                print(f"No data found in {sheet}.")
            else:
                data = pd.DataFrame(values[1:], columns=values[0])
                dfs[sheet] = data
        except HttpError as err:
            print(f"Error for data from {sheet}: {err}")

    ti = kwargs['ti']
    ti.xcom_push(key='dfs', value=dfs)


def clean_data(**kwargs):
    ti = kwargs['ti']
    dfs = ti.xcom_pull(task_ids='load_data', key='dfs')

    for sheet, data in dfs.items():
        data = data.dropna()
        data = data.drop_duplicates()
        dfs[sheet] = data

    ti.xcom_push(key='dfs', value=dfs)


def feature_engineering(**kwargs):
    ti = kwargs['ti']
    dfs = ti.xcom_pull(task_ids='clean_data', key='dfs')

    for sheet, data in dfs.items():
        data['message_length'] = data['message'].apply(len)
        data['num_digits'] = data['message'].apply(lambda x: sum(c.isdigit() for c in x))
        data['num_uppercase'] = data['message'].apply(lambda x: sum(c.isupper() for c in x))
        data['num_special_chars'] = data['message'].apply(lambda x: sum(not c.isalnum() for c in x))
        data['num_words'] = data['message'].apply(lambda x: len(x.split()))
        data['avg_word_length'] = data['message'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0)

        dfs[sheet] = data

    ti.xcom_push(key='dfs', value=dfs)


def scale_and_normalize(**kwargs):
    ti = kwargs['ti']
    dfs = ti.xcom_pull(task_ids='feature_engineering', key='dfs')

    for sheet, data in dfs.items():
        for col in data.select_dtypes(include=[np.number]).columns:
            mean = data[col].mean()
            std = data[col].std()
            data[col] = (data[col] - mean) / std

        for col in data.select_dtypes(include=[np.number]).columns:
            min_val = data[col].min()
            max_val = data[col].max()
            data[col] = (data[col] - min_val) / (max_val - min_val)

        dfs[sheet] = data

    ti.xcom_push(key='dfs', value=dfs)


def save_to_google_sheets(**kwargs):
    ti = kwargs['ti']
    dfs = ti.xcom_pull(task_ids='feature_engineering', key='dfs')

    for sheet, data in dfs.items():
        print(f"Saving to Google Sheets: {sheet} with columns {data.columns.tolist()}")

    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_PATH, scope)
    client = gspread.authorize(creds)

    train_sheet = client.open("spam_train").sheet1
    test_sheet = client.open("spam_test").sheet1

    for sheet_name, data in dfs.items():
        print(f"Uploading {sheet_name} with {len(data)} rows and columns: {data.columns.tolist()}")

        if sheet_name == 'spam_train':
            sheet = train_sheet
        elif sheet_name == 'spam_test':
            sheet = test_sheet
        else:
            continue

        values = data.values.tolist()
        sheet.update([data.columns.values.tolist()] + values)
        print(f"Data uploded to Google Sheets: {sheet_name}")

    print("Data saved to Google Sheets.")


default_args = {
    'owner': 'piotr',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1),
}

dag = DAG(
    'process_data_dag',
    default_args=default_args,
    description='DAG for processing spam data from Google Sheets',
    schedule_interval=None,
)

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data_from_google_sheets,
    provide_context=True,
    dag=dag,
)

clean_data_task = PythonOperator(
    task_id='clean_data',
    python_callable=clean_data,
    provide_context=True,
    dag=dag,
)

feature_engineering_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_engineering,
    provide_context=True,
    dag=dag,
)

scale_and_normalize_task = PythonOperator(
    task_id='scale_and_normalize',
    python_callable=scale_and_normalize,
    provide_context=True,
    dag=dag,
)

save_to_google_sheets_task = PythonOperator(
    task_id='save_to_google_sheets',
    python_callable=save_to_google_sheets,
    provide_context=True,
    dag=dag,
)

load_data_task >> clean_data_task >> feature_engineering_task >> scale_and_normalize_task >> save_to_google_sheets_task
