import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, mixture

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

def plot(data, label, prefix):
    x = data[0]
    y = data[1]
    color = ['r', 'g', 'b']
    for i in xrange(3):
        plt.plot(x[label == i], y[label == i], color[i] + 'o')
    plt.savefig('result/' + prefix + '.jpg')

def print_result(res_file, a, name):
    res_file.write(name + ':\n')
    for i in xrange(len(a)):
        s = str(a[i]) if a.ndim == 1 else ' '.join(map(str, a[i]))
        res_file.write(s + '\n')

def kmeans_print_result(classifier, prefix):
    res_file = open('result/' + prefix + '.txt', 'w')
    print_result(res_file, classifier.cluster_centers_, 'center')
    res_file.close()

def gmm_print_result(classifier, prefix):
    res_file = open('result/' + prefix + '.txt', 'w')
    print_result(res_file, classifier.weights_, 'weight')
    print_result(res_file, classifier.means_, 'mean')
    print_result(res_file, classifier.covars_, 'covariance')
    res_file.close()

def main(classifier, print_result, prefix):
    raw_data = get_data('data.csv')
    data = np.ndarray(shape = (2, len(raw_data)))
    data[0] = get_field(raw_data, 'V1')
    data[1] = get_field(raw_data, 'V2')
    data = data.T
    classifier.fit(data)
    plot(data.T, classifier.predict(data), prefix)
    print_result(classifier, prefix)

np.random.seed(19930131)
main(cluster.KMeans(n_clusters = 3), kmeans_print_result, 'kmeans')
main(mixture.GMM(n_components = 3), gmm_print_result, 'gmm')
