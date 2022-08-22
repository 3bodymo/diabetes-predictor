import warnings
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd


def warn(*args, **kwargs):
    pass


warnings.warn = warn


def main():
    data = pd.read_csv('./datasets/dataset.csv')
    error = ['Glucose', 'BloodPressure', 'SkinThickness',
             'Insulin', 'BMI', 'DiabetesPedigreeFunction']

    data[error] = data[error].replace(0, np.NaN)
    si = SimpleImputer(missing_values=np.NaN, strategy="mean")
    data[error] = si.fit_transform(data[error])
    data_major = data[(data['Outcome'] == 0)]
    data_minor = data[(data['Outcome'] == 1)]
    upsample = resample(data_minor,
                        replace=True,
                        n_samples=500,
                        random_state=42)
    data = pd.concat([upsample, data_major])
    sample_data = data.drop('Outcome', axis=1)
    sample_data_results = data['Outcome']
    train_sample_data, test_sample_data, train_sample_data_results, test_sample_data_results = train_test_split(
        sample_data, sample_data_results, test_size=0.3, random_state=42)
    gnb = GaussianNB()
    model = gnb.fit(train_sample_data, train_sample_data_results)
    return model


def prediction(model, tp, pg, dbp, tst, si, bmi, dpf, age):
    preds = model.predict([[tp, pg, dbp, tst, si, bmi, dpf, age]])
    return preds
