from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def get_model(name):
    if name == 'logistic_regression':
        return LogisticRegression()
    elif name == 'naive_bayes':
        return GaussianNB()
    elif name == 'decision_tree':
        return DecisionTreeClassifier()
    elif name == 'SVM':
        return SVC(kernel = 'poly')
    else:
        raise ValueError('No such model')
