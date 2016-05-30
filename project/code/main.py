import data
import model
import numpy as np
import random

random.seed(19930131)
np.random.seed(19930131)

def training(model_name):
    feature, label = data.get_matrix('train')
    classifier = model.get_model(model_name)
    classifier.fit(feature, label)
    return classifier

def testing(classifier):
    feature, label = data.get_matrix('test')
    return classifier.score(feature, label) * 100.0

for model_name in ('logistic regression', 'naive bayes', 'decision tree', 'SVM'):
    classifier = training(model_name)
    print model_name, testing(classifier)
