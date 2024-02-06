# https://docs.deepchecks.com/monitoring/stable/user-guide/auto_demos/plot_lending_defaults.html
# monitoring.py
# creating a client
import os
from deepchecks_client import DeepchecksClient

token = {token}
host = 'http://localhost'

dc_client = DeepchecksClient(host=host, token=token)

# getting the data
import pandas as pd

train_df = pd.read_csv('https://figshare.com/ndownloader/files/39316160')

# data schema
from deepchecks.tabular import Dataset
from deepchecks_client import create_schema, read_schema

features = ['sub_grade', 'term', 'home_ownership', 'fico_range_low',
            'total_acc', 'pub_rec', 'revol_util', 'annual_inc', 'int_rate', 'dti',
            'purpose', 'mort_acc', 'loan_amnt', 'application_type', 'installment',
            'verification_status', 'pub_rec_bankruptcies', 'addr_state',
            'initial_list_status', 'fico_range_high', 'revol_bal', 'open_acc',
            'emp_length', 'time_to_earliest_cr_line']
cat_features = ['sub_grade', 'home_ownership', 'term', 'purpose', 'application_type', 'verification_status', 'addr_state',
                'initial_list_status']

dataset_kwargs = {
    'features': features,
    'cat_features': cat_features,
    'index_name': 'id',
    'label': 'loan_status',
    'datetime_name': 'issue_d'
}

train_dataset = Dataset(train_df, **dataset_kwargs)

schema_file_path = 'schema_file.yaml'
create_schema(dataset=train_dataset, schema_output_file=schema_file_path)
read_schema(schema_file_path)

# feature importance
import joblib
from urllib.request import urlopen

with urlopen('https://figshare.com/ndownloader/files/39316172') as f:
    model = joblib.load(f)

feature_importance = pd.Series(model.feature_importances_ / sum(model.feature_importances_), index=model.feature_names_)

# creating a model version
ref_predictions = model.predict(train_df[features])
ref_predictions_proba = model.predict_proba(train_df[features])

model_name = 'Loan Defaults - Example'

model_version = dc_client.create_tabular_model_version(model_name=model_name, version_name='ver_1',
                                                       schema=schema_file_path,
                                                       feature_importance=feature_importance,
                                                       reference_dataset=train_dataset,
                                                       reference_predictions=ref_predictions,
                                                       reference_probas=ref_predictions_proba,
                                                       task_type='binary',
                                                       model_classes=[0, 1],
                                                       monitoring_frequency='month')

# uploading production data
prod_data = pd.read_csv('https://figshare.com/ndownloader/files/39316157', parse_dates=['issue_d'])

import datetime

time_delta = pd.Timedelta(pd.to_datetime(datetime.datetime.now()) - prod_data['issue_d'].max()) - pd.Timedelta(2, unit='d')
prod_data['issue_d'] = prod_data['issue_d'] + time_delta
prod_data['issue_d'].unique()

# uploading a batch of data
prod_predictions = model.predict(prod_data[train_dataset.features].fillna('NONE'))
prod_prediction_probas = model.predict_proba(prod_data[train_dataset.features].fillna('NONE'))

model_version.log_batch(sample_ids=prod_data['id'],
                        data=prod_data.drop(['issue_d', 'id', 'loan_status'], axis=1),
                        timestamps=(prod_data['issue_d'].astype(int) // 1e9).astype(int),
                        predictions=prod_predictions,
                        prediction_probas=prod_prediction_probas)

# uploading the labels
model_client = dc_client.get_or_create_model(model_name)
model_client.log_batch_labels(sample_ids=prod_data['id'], labels=prod_data['loan_status'])
