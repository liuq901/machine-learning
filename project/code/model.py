from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def get_model(name):
    if name == 'logistic regression':
        return LogisticRegression()
    elif name == 'naive bayes':
        return GaussianNB()
    elif name == 'decision tree':
        return DecisionTreeClassifier()
    elif name == 'SVM':
        return SVC(kernel = 'poly')
    else:
        raise ValueError('No such model')
