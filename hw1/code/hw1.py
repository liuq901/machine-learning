import csv
import numpy as np
import matplotlib.pyplot as plt

def get_data(file_name):
    csv_file = open('data/' + file_name, 'r')
    reader = csv.DictReader(csv_file)
    data = []
    for row in reader:
        data.append(row)
    csv_file.close()
    return data

def normalize(data):
    mean = data.mean()
    std = data.std()
    data = (data - mean) / std
    return data, mean, std

def get_field(data, field):
    res = []
    for d in data:
        res.append(float(d[field]))
    return np.array(res)

def gradient_descent(feature, label):
    alpha = 1e-1 / feature.shape[0]
    parameter = np.zeros(feature.shape[1])
    for i in xrange(1000):
        delta = feature.dot(parameter) - label
        gradient = feature.T.dot(delta)
        parameter -= alpha * gradient
    return parameter

def newtons_method(feature, label):
    parameter = np.zeros(feature.shape[1])
    for i in xrange(10):
        delta = feature.dot(parameter) - label
        gradient = feature.T.dot(delta)
        hessian = feature.T.dot(feature)
        parameter -= np.linalg.inv(hessian).dot(gradient)
    return parameter

def normal_equation(feature, label):
    tmp = np.linalg.inv(feature.T.dot(feature))
    return tmp.dot(feature.T).dot(label)

def resume(parameter, mean, std, label_mean, label_std):
    n = len(mean)
    for i in xrange(n):
        parameter[i] /= std[i]
        parameter[n] -= mean[i] * parameter[i]
    parameter *= label_std
    parameter[n] += label_mean
    return parameter

def training(data, feature_name, label_name):
    res = []
    for method in [gradient_descent, newtons_method, normal_equation]:
        n = len(feature_name)
        m = len(data)
        feature = np.ndarray(shape = (n + 1, m), dtype = float)
        mean, std = [None] * n, [None] * n
        for i in xrange(n):
            feature[i], mean[i], std[i] = normalize(get_field(data, feature_name[i]))
        feature[n] = np.ones(m)
        feature = feature.T
        label, label_mean, label_std = normalize(get_field(data, label_name))
        parameter = resume(method(feature, label), mean, std, label_mean, label_std)
        res.append(parameter)
    return res

def get_line(xmin, xmax, line):
    x = np.array([xmin, xmax])
    y = x * line[0] + line[1]
    return x, y

def plot(x, y, line, x_label, y_label, prefix):
    plt.clf()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x, y, 'oy')
    linex, liney = get_line(x.min(), x.max(), line)
    plt.plot(linex, liney, 'r')
    plt.savefig('result/' + prefix + '.jpg')

def get_result(data, parameter, feature_name, label_name, do_plot, print_result, prefix):
    method_name = ['gradient_descent', 'newtons_method', 'normal_equation']
    for i in xrange(len(parameter)):
        n = len(feature_name)
        m = len(data)
        feature = np.ndarray(shape = (n + 1, m), dtype = float)
        for j in xrange(n):
            feature[j] = get_field(data, feature_name[j])
        feature[n] = np.ones(m)
        label = get_field(data, label_name)
        if do_plot:
            plot(feature[0], label, parameter[i], feature_name[0], label_name, prefix + '_' + method_name[i])
        if print_result:
            res_file = open('result/' + prefix + '_' + method_name[i] + '.txt', 'w')
            res_file.write('field: ' + ' '.join(feature_name) + '\n')
            res_file.write('parameter: ' +  ' '.join(map(str, parameter[i])) + '\n')
            res_file.write('rmse: ' + str(np.sqrt(((feature.T.dot(parameter[i]) - label) ** 2).mean())) + '\n')
            res_file.close()

def main(plot, prefix, *feature_name):
    train_data = get_data('train.csv')
    test_data = get_data('test.csv')
    label_name = 'price'
    parameter = training(train_data, feature_name, label_name)
    if plot:
        get_result(train_data, parameter, feature_name, label_name, True, False, prefix + '_train')
        get_result(test_data, parameter, feature_name, label_name, True, True, prefix + '_test')
    else:
        get_result(test_data, parameter, feature_name, label_name, False, True, prefix + '_test')

main(True, 'required', 'sqft_living')
main(False, 'optional', 'sqft_living', 'bedrooms')
