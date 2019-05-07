import pandas as pd
import numpy as np
from sklearn.utils import shuffle as sklearn_shuffle


def load_dataset():
    data = pd.read_csv('dataset/data.csv',sep=',', header=None)
    data.columns=['messages', 'labels']
    return (*split_dataset(data), data)

def split_dataset(data):
    dataset = data.copy(deep=True)
    labels = dataset['labels'].unique()
    test_datasets = pd.DataFrame()
    train_datasets = pd.DataFrame()

    for label in labels:
        ds_with_label = dataset[dataset['labels'] == label]
        num_test = int(ds_with_label.shape[0]*0.2)
        num_test = 1 if num_test < 1 and ds_with_label.shape[0] > 1 else num_test
        ds_test = ds_with_label[:num_test]
        ds_train = ds_with_label[num_test:]
        test_datasets=pd.concat((test_datasets, ds_test))
        train_datasets=pd.concat((train_datasets, ds_train))

    test_datasets = sklearn_shuffle(test_datasets, random_state=0)
    train_datasets = sklearn_shuffle(train_datasets, random_state=0)

    train_messages = np.array(train_datasets['messages']).reshape(train_datasets.shape[0], 1)
    train_labels = np.array(train_datasets['labels']).reshape(train_datasets.shape[0], 1)
    test_messages = np.array(test_datasets['messages']).reshape(test_datasets.shape[0], 1)
    test_labels = np.array(test_datasets['labels']).reshape(test_datasets.shape[0], 1)

    return train_messages, train_labels, test_messages, test_labels

def split_dataset_data_frame(data):
    dataset = data.copy(deep=True)
    labels = dataset['labels'].unique()
    test_datasets = pd.DataFrame()
    train_datasets = pd.DataFrame()

    for label in labels:
        ds_with_label = dataset[dataset['labels'] == label]
        num_test = int(ds_with_label.shape[0]*0.2)
        num_test = 1 if num_test < 1 and ds_with_label.shape[0] > 1 else num_test
        ds_test = ds_with_label[:num_test]
        ds_train = ds_with_label[num_test:]
        test_datasets=pd.concat((test_datasets, ds_test))
        train_datasets=pd.concat((train_datasets, ds_train))

    test_datasets = sklearn_shuffle(test_datasets, random_state=0)
    train_datasets = sklearn_shuffle(train_datasets, random_state=0)

    return train_datasets, test_datasets
