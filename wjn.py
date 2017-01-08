import numpy as np
import pickle
from analysis_morphologic import create_datasets, WjnContents
from datasets import base
import os.path
import random


class Dataset(object):
    def __init__(self,
                 dicts,
                 labels):

        self._num_examples = len(dicts)
        self._dicts = dicts
        self._dict_size = dicts.shape[1]
        self._labels = labels
        self._index_in_epoch = 0

    @property
    def dicts(self):
        return self._dicts

    @property
    def labels(self):
        return self._labels

    @property
    def dict_size(self):
        return self._dict_size

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            self._dicts, self._labels = \
                self._shuffle([self._dicts, self._labels], self._num_examples)
            self._index_in_epoch = batch_size
            start = 0
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._dicts[start:end], self._labels[start:end]

    def _shuffle(self, all_list, num):
        perm = np.arange(num)
        np.random.shuffle(perm)
        for i in all_list:
            yield [i[j] for j in perm]


def read_data_sets(test_size=60,
                   validation_size=0):

    if os.path.exists(base.fn_datasets % (test_size, validation_size)):
        with open(base.fn_datasets % (test_size, validation_size), 'rb') as f:
            train, validation, test = pickle.load(f)

    else:
        wjn = WjnContents()
        wjn_array = np.array([wjn.contents, wjn.labels])
        inv_wjn_array = np.transpose(wjn_array)
        inv_wjn_list = inv_wjn_array.tolist()
        random.shuffle(inv_wjn_list)

        test_data = inv_wjn_list[:test_size]
        validation_data = inv_wjn_list[test_size:test_size+validation_size]
        train_data = inv_wjn_list[test_size+validation_size:]

        test_data = np.transpose(test_data)
        validation_data = np.transpose(validation_data)
        train_data = np.transpose(train_data)

        train_dicts, train_labels = create_datasets(
            train_data[0], train_data[1], create_dict=True)
        test_dicts, test_labels = create_datasets(test_data[0], test_data[1])
        if validation_size > 0:
            validation_dicts, validation_labels = create_datasets(
                validation_data[0], validation_data[1])
        else:
            validation_dicts = np.array([[]])
            validation_labels = np.array([[]])

        train = Dataset(train_dicts, train_labels)
        validation = Dataset(validation_dicts, validation_labels)
        test = Dataset(test_dicts, test_labels)

        data = [train, validation, test]

        with open(base.fn_datasets % (test_size, validation_size), 'wb') as f:
            pickle.dump(data, f)
    print(test.labels)

    return base.Datasets(train=train, validation=validation, test=test)
