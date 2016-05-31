import numpy as np
import os
import random

feature_length = None

def set_feature_length(x):
    global feature_length
    feature_length = x

def read_file(file_name):
    tmp = []
    f = open(file_name, 'r')
    for x in f:
        tmp.append(float(x))
    f.close()
    return np.array(tmp)

def get_data(dataset):
    res = []
    subdir = ('one', 'two', 'three', 'four')
    for i in xrange(len(subdir)):
        path = 'data/' + dataset + '/' + subdir[i]
        for file_name in os.listdir(path):
            name, ext = os.path.splitext(file_name)
            if ext != '.engy':
                continue
            tmp = {}
            tmp['name'] = name
            tmp['engy'] = read_file(path + '/' + name + ext)
            tmp['f0'] = read_file(path + '/' + name + '.f0')
            tmp['label'] = i
            res.append(tmp)
    return res

def get_index(data):
    start = None
    length = 0
    for i in xrange(len(data)):
        if data[i] != 0:
            if not start:
                start = i
            length += 1
    assert feature_length <= length
    ratio = float(length) / float(feature_length)
    res = []
    for i in xrange(feature_length):
        res.append(int(i * ratio) + start)
    return res

def extract(data, index):
    res = np.ndarray(shape = feature_length)
    for i in xrange(feature_length):
        res[i] = data[index[i]]
    return res

def get_matrix(dataset, shuffle):
    data = get_data(dataset)
    if shuffle:
        random.shuffle(data)
    feature = np.ndarray(shape = (len(data), feature_length * 2))
    label = np.ndarray(shape = len(data))
    name = []
    for i in xrange(len(data)):
        index = get_index(data[i]['f0'])
        feature[i][0:feature_length] = extract(data[i]['engy'], index)
        feature[i][feature_length:] = extract(data[i]['f0'], index)
        label[i] = data[i]['label']
        name.append(data[i]['name'])
    return feature, label, name
