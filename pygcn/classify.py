from __future__ import print_function
import os
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import pdb
import numpy
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from time import time

class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)


class Classifier(object):

    def __init__(self, vectors, clf):
        self.embeddings = vectors
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)
        averages = ["micro", "macro", "samples", "weighted", "acc"]
        results = {}
        for average in averages:
            if average == "acc":
                results[average] = accuracy_score(Y, Y_)
            else:
                results[average] = f1_score(Y, Y_, average=average)
        # print('Results, using embeddings of dimensionality', len(self.embeddings[X[0]]))
        # print('-------------------')
        # print(results)
        return results
        # print('-------------------')

    def predict(self, X, top_k_list):
        X_ = numpy.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_precent, seed=0):
        state = numpy.random.get_state()

        training_size = int(train_precent * len(X))
        numpy.random.seed(seed)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y)
        numpy.random.set_state(state)
        return self.evaluate(X_test, Y_test)

def classify(embedding, Y, train_percent):
    vectors = {}
    for i in range(embedding.shape[0]):
        vectors[str(i)] = embedding[i]

    print("Training classifier using {:.2f}% nodes...".format(train_percent * 100))
    clf = Classifier(vectors=vectors, clf=LogisticRegression(solver="lbfgs", max_iter=4000))
    scores = clf.split_train_evaluate([str(i) for i in range(embedding.shape[0])], [[str(i)] for i in Y], train_percent)
    return scores
