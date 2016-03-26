#!/usr/bin/env python
# encoding: utf-8

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics


def test():
    corpus = []
    for line in open('tcorpus'):
        corpus.append(line.strip())
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)
    k_model = KMeans(n_clusters=3)
    k_model.fit(matrix)
    matrix = matrix.toarray()
    labels = k_model.labels_
    sil = metrics.silhouette_samples(matrix, labels)
    return sil, matrix, labels, k_model
