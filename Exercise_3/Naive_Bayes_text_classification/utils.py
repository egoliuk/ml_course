import pandas as pd
from sklearn.utils import shuffle as sklearn_shuffle


def load_dataset():
    data = pd.read_csv('dataset/data.csv',sep=',', header=None)
    data.columns=['message', 'labels']
    return (*split_dataset(data), data)

def split_dataset(data):
    dataset = data.copy(deep=True)
    labels = dataset['labels'].unique()
    test_datasets = pd.DataFrame()
    train_datasets = pd.DataFrame()

    for label in labels:
        ds_with_label = dataset[dataset['labels'] == label]
        num_test = int(ds_with_label.shape[0]*0.2)
        ds_test = ds_with_label[:num_test]
        ds_train = ds_with_label[num_test:]
        test_datasets=pd.concat((test_datasets, ds_test))
        train_datasets=pd.concat((train_datasets, ds_train))

    test_datasets = sklearn_shuffle(test_datasets, random_state=0)
    train_datasets = sklearn_shuffle(train_datasets, random_state=0)

    return train_datasets, test_datasets
