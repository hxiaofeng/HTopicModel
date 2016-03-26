#!/usr/bin/env python
# encoding: utf-8

from gensim import corpora
from gensim import matutils
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import kneighbors_graph
import numpy
import pickle


def k_means(matrix_fn, n_clusters=2):
    matrix = corpora.MmCorpus(matrix_fn)
    matrix = matutils.corpus2csc(matrix, num_terms=matrix.num_terms).transpose()
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(matrix)
    return matrix, kmeans.labels_


def hierarchical_clustering(corpus_fn, n_clusters=2, linkage='ward'):
    corpus = corpora.MmCorpus(corpus_fn)
    corpus = matutils.corpus2csc(corpus, num_terms=corpus.num_terms).transpose()
    svd = TruncatedSVD(n_components=100)
    new_corpus = svd.fit_transform(corpus)
    knn_graph = kneighbors_graph(new_corpus, 100, metric='euclidean')
    agg = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage=linkage, connectivity=knn_graph)
    agg.fit(new_corpus)
    return corpus, agg.labels_


def k_means_main():
    corpus_fn = '../data/models/sougou_tfidf_matrix.mm'
    matrix, labels = k_means(corpus_fn, 400)
    pickle.dump(labels, open('../data/compare/n_400_labels', 'wb'))
    scores = []
    for i in range(10):
        score = metrics.silhouette_score(matrix, labels, sample_size=5000, n_jobs=10)
        scores.append(score)
        print score
    sc = numpy.array(scores).mean()
    print 'mean', sc


def hie_main():
    corpus_fn = '../data/models/sougou_tfidf_matrix.mm'
    matrix, labels = hierarchical_clustering(corpus_fn, 400, 'ward')
    pickle.dump(labels, open('../data/compare/hie_ward_n_400_labels', 'wb'))
    scores = []
    for i in range(10):
        score = metrics.silhouette_score(matrix, labels, metric='euclidean', sample_size=5000, n_jobs=10)
        scores.append(score)
        print score
    sc = numpy.array(scores).mean()
    print 'mean', sc


if __name__ == '__main__':
    hie_main()
