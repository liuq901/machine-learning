import data
import model
import numpy as np
import random

random.seed(19930131)
np.random.seed(19930131)

def training(model_name):
    feature, label, _ = data.get_matrix('train', True)
    classifier = model.get_model(model_name)
    classifier.fit(feature, label)
    return classifier

def testing(prefix, classifier):
    feature, label, name = data.get_matrix('test', False)
    predict = classifier.predict(feature)
    res_file = open(prefix + '.txt', 'w')
    res_file.write(str(classifier.score(feature, label) * 100.0) + '\n')
    for i in xrange(4):
        res_file.write(' '.join(map(str, (np.logical_and(label == i, predict == j).sum() for j in xrange(4)))) + '\n')
    res_file.write('predict label name\n')
    for x in zip(predict, label, name):
        res_file.write(' '.join(map(str, x)) + '\n')
    res_file.close()

for length in (20, 15, 10, 5):
    data.set_feature_length(length)
    for model_name in ('logistic_regression', 'naive_bayes', 'decision_tree', 'SVM'):
        classifier = training(model_name)
        testing('result/' + model_name + '_' + str(length), classifier)
