import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn import svm

def get_data(file_name):
    csv_file = open('data/' + file_name, 'r')
    reader = csv.DictReader(csv_file)
    data = []
    for row in reader:
        data.append(row)
    csv_file.close()
    return data

def get_field(data, field):
    res = []
    for d in data:
        res.append(float(d[field]))
    return np.array(res)

def get_feature_names(names):
    pattern = 'word'
    res = []
    for name in names:
        if re.match(pattern, name):
            res.append(name)
    return res

def get_matrix(data, feature_names):
    res = np.ndarray(shape = (len(feature_names), len(data)))
    for i in xrange(len(feature_names)):
        name = feature_names[i]
        res[i] = get_field(data, name)
    return res.T

def get_array(file_name):
    data = get_data(file_name)
    feature_names = get_feature_names(data[0].iterkeys())
    label_name = 'is_spam'
    feature = get_matrix(data, feature_names)
    label = get_field(data, label_name)
    return feature, label

def spam_filter():
    for kernel in ['linear', 'poly', 'rbf']:
        feature, label = get_array('train.csv')
        clf = svm.SVC(kernel = kernel)
        clf.fit(feature, label)
        feature, label = get_array('test.csv')
        predict = clf.predict(feature)
        spam = 1
        accuracy = (predict == label).sum() * 1.0 / len(feature)
        spam_right = np.logical_and(predict == spam, label == spam)
        precision = spam_right.sum() * 1.0 / (predict == spam).sum()
        recall = spam_right.sum() * 1.0 / (label == spam).sum()
        result_file = open('result/' + kernel + '.txt', 'w')
        result_file.write('accuracy: ' + str(accuracy) + '\n')
        result_file.write('precision: ' + str(precision) + '\n')
        result_file.write('recall: ' + str(recall) + '\n')
        result_file.close()

def plot(positive_x, positive_y, negative_x, negative_y, boundary, name):
    plt.figure(figsize = (5, 5))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-3.5, 3.5)
    plt.ylim(-3.5, 3.5)
    plt.plot(positive_x, positive_y, 'ro')
    plt.plot(negative_x, negative_y, 'bx')
    boundary()
    plt.savefig('result/' + name + '.jpg')

def linear_boundary():
    plt.plot([-3.5, 3.5], [3.5, -3.5], 'y')

def circular_boundary():
    circle = Circle((0, 0), radius = 2, color = 'y', fill = False)
    plt.gca().add_patch(circle)

def boundary_plot():
    positive_x = [[-2, -2, -1, -1], [-3, 3, 0, 0]]
    positive_y = [[-2, -1, -2, -1], [0, 0, -3, 3]]
    negative_x = [[2, 2, 1, 1], [-1, 1, 0, 0]]
    negative_y = [[2, 1, 2, 1], [0, 0, -1, 1]]
    boundary = [linear_boundary, circular_boundary]
    name = ['linear_boundary', 'circular_boundary']
    for i in xrange(len(positive_x)):
        plot(positive_x[i], positive_y[i], negative_x[i], negative_y[i], boundary[i], name[i])

def main():
    spam_filter()
    boundary_plot()

main()
