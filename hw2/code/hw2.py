import csv
import numpy as np
from scipy.stats import norm

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

def main(prefix, *removed):
    train_data = get_data('train.csv')
    test_data = get_data('test.csv')
    label_name = 'is_spam'
    label_count = 2
    label = get_field(train_data, label_name)
    removed = removed + (label_name, )
    confidence = np.zeros((label_count, len(test_data)))
    for key in train_data[0].iterkeys():
        flag = False
        for pattern in removed:
            if key == pattern:
                flag = True
        if flag:
            continue
        train_field = get_field(train_data, key)
        test_field = get_field(test_data, key)
        std = train_field.std()
        for i in xrange(label_count):
            mean = train_field[label == i].mean()
            likelihood = norm(mean, std).pdf(test_field)
            confidence[i] += np.log(likelihood + 1e-26)
    for i in xrange(label_count):
        prior = (label == i).sum() * 1.0 / len(train_data)
        confidence[i] += np.log(prior)
    predict = confidence.argmax(axis = 0)
    label = get_field(test_data, label_name)
    accuracy = (predict == label).sum() * 1.0 / len(test_data)
    result = open('result/' + prefix + '.txt', 'w')
    result.write('removed: ' + ' '.join(removed) + '\n')
    result.write('accuracy: ' + str(accuracy) + '\n')

main('required')
main('optional', 'word_freq_make')
