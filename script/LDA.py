#!/usr/bin/env python
# encoding: utf-8

from gensim import corpora
from gensim import models
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy
import terms_analysis
import pickle


# 对语料库的tfidf矩阵进行LDA主题分析
def LDA_analysis(corpus_file, dictionary_file, out_model_file, num_topics=20):
    corpus = corpora.MmCorpus(corpus_file)
    dictionary = corpora.Dictionary.load(dictionary_file)
    lda = models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, workers=20)
    lda.save(out_model_file)


# 获取主题词项集，即将每个词项归为主题分布值最高的那个主题，从而使每个主题对应一个词项集
def get_topics_terms(lda_model_file):
    lda = models.LdaMulticore.load(lda_model_file)
    topics = lda.show_topics(num_topics=lda.num_topics, num_words=lda.num_terms, formatted=False)
    term2topics = dict()
    for i in range(len(topics)):
        topic = topics[i]
        for pro, term in topic:
            term2topics.setdefault(term, []).append((i, pro))
    topic2terms = dict()
    for term, topics in term2topics.iteritems():
        max_topic = max(topics, key=lambda x: x[1])[0]
        topic2terms.setdefault(max_topic, []).append(term)
    return topic2terms.items()


# 获取词项-主题矩阵，矩阵的每一行表示一个词项，一列表示一个主题
def get_term_topic_matrix(lda_model_filename):
    lda = models.LdaMulticore.load(lda_model_filename)
    term2vec = dict()
    for term in lda.id2word.values():
        term2vec[term] = []
    for i in range(lda.num_topics):
        topic = lda.show_topic(i, topn=lda.num_terms)
        for v, t in topic:
            term2vec[t].append(v)
    matrix = []
    dictionary = []
    for term, vec in term2vec.iteritems():
        if len(vec) == lda.num_topics:
            dictionary.append(term)
            matrix.append(vec)
    matrix = numpy.array(matrix)
    return dictionary, matrix


# 对词项-主题矩阵进行聚类分析
def terms_clustering(lda_model_filename, clusters_out_filename, n_clusters=2):
    dictionary, matrix = get_term_topic_matrix(lda_model_filename)
    clusters = terms_analysis.cluster(matrix, dictionary, n_clusters, 10)
    pickle.dump(clusters, open(clusters_out_filename, 'wb'))


def n_analysis():
    _, matrix = get_term_topic_matrix('../data/models/sougou_lda_100_model.md')
    for i in range(2, 101):
        model = KMeans(i, n_jobs=10)
        model.fit(matrix)
        sc_array = []
        for j in range(30):
            sc = metrics.silhouette_score(matrix, model.labels_, sample_size=10000, n_jobs=10)
            sc_array.append(sc)
        sc = numpy.array(sc_array).mean()
        print i, sc


def sougou_main():
    LDA_analysis('../data/models/sougou_tfidf_matrix.mm',
            '../data/models/sougou.dict',
            '../data/models/sougou_lda_20_model.md',
            20)


if __name__ == '__main__':
    # LDA_analysis('../data/wiki_models/200k/wiki_tfidf_matrix.mm',
                 # '../data/wiki_models/200k/wiki.dict',
                 # '../data/wiki_models/200k/wiki_lda_20_model.md',
    #              20)
    sougou_main()
   #  terms_clustering('../data/models/sougou_lda_100_model.md',
            # '../data/lda_clusters/sougou_lda_100_n_12',
   #          12)
    # n_analysis()
