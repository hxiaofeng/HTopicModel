#!/usr/bin/env python
# encoding: utf-8

import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from gensim import models
import LDA
import utilities


# 获取词集合的词向量矩阵，矩阵的每一行对应一个词
def get_words_matrix(words, word2vec_model):
    matrix = []
    dictionary = []
    for word in words:
        try:
            vec = word2vec_model[word]
            matrix.append(vec.tolist())
            dictionary.append(word)
        except KeyError:
            pass
    matrix = np.array(matrix)
    return dictionary, matrix


# 对词集进行聚类
def cluster(matrix, dictionary, n_clusters=2, n_jobs=1):
    cls_model = KMeans(n_clusters=n_clusters, n_jobs=n_jobs)
    cls_model.fit(matrix)
    clusters = dict()
    labels = cls_model.labels_
    for i in range(len(dictionary)):
        clusters.setdefault(labels[i], []).append(dictionary[i])
    centers = dict()
    for i in range(len(cls_model.cluster_centers_)):
        centers[i] = cls_model.cluster_centers_[i]
    return clusters, centers


# 对lda的每个主题下的词项进行二分聚类，过滤无用的词
def lda_terms_analysis(lda_model_filename, word2vec_model_filename):
    topics = LDA.get_topics_terms(lda_model_filename)
    word2vec = models.Word2Vec.load(word2vec_model_filename)
    new_topics = []
    useless = []
    for topic in topics:
        words = topic[-1]
        dictionary, matrix = get_words_matrix(words, word2vec)
        clusters, centers = cluster(matrix, dictionary, 2, 10)
        cohesions = []
        for c in clusters.items():
            sub_words = c[-1]
            label = c[0]
            _, sub_matrix = get_words_matrix(sub_words, word2vec)
            center = centers[label]
            cohesion = utilities.cohesion(sub_matrix, center)
            cohesions.append((label, cohesion))
        cohesions.sort(key=lambda x: x[-1])
        new_topic = list(topic[:-1])
        new_topic.append(cohesions[0][1])
        new_topic.append(clusters[cohesions[0][0]])
        new_topics.append(new_topic)
        for c in cohesions[1:]:
            u_topic = list(topic[:-1])
            u_topic.append(c[0])
            u_topic.append(c[1])
            u_topic.append(clusters[c[0]])
            useless.append(u_topic)
    return new_topics, useless


# 对lda分析生成的词集进行基于词向量的聚类分析
def hierarchical_topic_analyse(lda_model_file, word2vec_model_file, k=1):
    topic2terms = LDA.get_topics_terms(lda_model_file)
    topics = []
    for t in topic2terms:
        topics.append(list(t))
    word2vec_model = models.Word2Vec.load(word2vec_model_file)
    for i in range(k):
        new_topics = []
        for t in topics:
            words = t[-1]
            dictionary, matrix = get_words_matrix(words, word2vec_model)
            clusters = cluster(matrix, dictionary, 2, 10)
            for item in clusters:
                labels = t[:-1]
                labels.extend(list(item))
                new_topics.append(labels)
            del dictionary
            del matrix
            del clusters
        topics = new_topics
    # topic_tree = get_topic_tree(topics)
    # return topic_tree
    return topics


# 对lda生成的词项集进行基于词向量的聚类分析，使用轮廓系数控制聚类过程
def hierarchical_topic_analyse_with_silhouette(corpus_filename, word2vec_model_filename, lda_filter=False, k=1):
    if lda_filter:
        topic2terms = pickle.load(open(corpus_filename))
    else:
        topic2terms = LDA.get_topics_terms(corpus_filename)
        # topic2terms, _ = lda_terms_analysis(corpus_filename, word2vec_model_filename)
    topics = []
    for t in topic2terms:
        topics.append(list(t))
    if k == 0:
        return topics
    word2vec_model = models.Word2Vec.load(word2vec_model_filename)
    new_topics = []
    for topic in topics:
        words = topic[-1]
        clusters = cluster_analyse_with_silhouette(words, word2vec_model, k)
        for c in clusters:
            new_topic = topic[:-1]
            new_topic.extend(c)
            new_topics.append(new_topic)
    return new_topics


# 对词项集进行聚类分析，根据整个聚类结果的轮廓系数的增减来决定是否对一个簇进行分裂
def adp_cluster_analyse_with_silhouette(words, word2vec_model, k=1):
    dictionary, matrix = get_words_matrix(words, word2vec_model)
    clusters = cluster(matrix, dictionary, 2, 10)
    topics = []
    depth = 0
    for c in clusters:
        topics.append(list(c))
        if len(c) - 1 > depth:
            depth = len(c) - 1
    labels, samples_score = silhouette_samples(topics, word2vec_model)
    clusters_score = silhouette_clusters(labels, samples_score)
    clusters_score.sort(key=lambda x: x[1])
    score = samples_score.mean()
    while clusters_score and depth < k:
        label = clusters_score[0][0]
        del clusters_score[0]
        topic = topics[label]
        subwords = topic[-1]
        dictionary, matrix = get_words_matrix(subwords, word2vec_model)
        clusters = cluster(matrix, dictionary, 2, 10)
        new_topics = []
        for c in clusters:
            new_topic = topic[:-1]
            new_topic.extend(list(c))
            new_topics.append(new_topic)
        new_topics.extend(topics[:label])
        new_topics.extend(topics[label + 1:])
        new_labels, new_samples_score = silhouette_samples(new_topics, word2vec_model)
        new_score = new_samples_score.mean()
        if new_score >= score:
            clusters_score = silhouette_clusters(new_labels, new_samples_score)
            clusters_score.sort(key=lambda x: x[1])
            score = new_score
            topics = new_topics
        for t in topics:
            if len(t) -1 > depth:
                depth = len(t) - 1
    return topics


# 对词项集进行层次聚类分析，每次选择轮廓系数最小的簇进行分裂,不回溯
def cluster_analyse_with_silhouette(words, word2vec_model, k=1):
    dictionary, matrix = get_words_matrix(words, word2vec_model)
    clusters, _ = cluster(matrix, dictionary, 2, 10)
    topics = []
    depth = 0
    for c in clusters.items():
        topics.append(list(c))
        if len(c) - 1 > depth:
            depth = len(c) - 1
    labels, samples_score = silhouette_samples(topics, word2vec_model)
    clusters_score = silhouette_clusters(labels, samples_score)
    clusters_score.sort(key=lambda x: x[1])
    while clusters_score and depth < k:
        label = clusters_score[0][0]
        del clusters_score[0]
        topic = topics[label]
        subwords = topic[-1]
        dictionary, matrix = get_words_matrix(subwords, word2vec_model)
        clusters, _ = cluster(matrix, dictionary, 2, 10)
        new_topics = []
        for c in clusters.items():
            new_topic = topic[:-1]
            new_topic.extend(list(c))
            new_topics.append(new_topic)
        new_topics.extend(topics[:label])
        new_topics.extend(topics[label + 1:])
        new_labels, new_samples_score = silhouette_samples(new_topics, word2vec_model)
        clusters_score = silhouette_clusters(new_labels, new_samples_score)
        clusters_score.sort(key=lambda x: x[1])
        topics = new_topics
        for t in topics:
            if len(t) -1 > depth:
                depth = len(t) - 1
    return topics


# 对词项集进行层次聚类分析，每次选择凝聚度最大的簇进行分裂,不回溯
def cluster_analyse_with_cohesion(words, word2vec_model, k=1):
    dictionary, matrix = get_words_matrix(words, word2vec_model)
    clusters, centers = cluster(matrix, dictionary, 2, 10)
    topics = []
    t_centers = dict()
    depth = 0
    cohesions = []
    for c in clusters.items():
        topics.append(list(c))
        label = c[:-1]
        t_centers[label] = centers[c[0]]
        words = c[-1]
        sub_matrix = get_words_matrix(words, word2vec_model)
        center = t_centers[label]
        cohesion = utilities.cohesion(sub_matrix, center)
        cohesions.append((label, cohesion))
        if len(c) - 1 > depth:
            depth = len(c) - 1
    cohesions.sort(key=lambda x: x[-1])
    while clusters_score and depth < k:
        label = clusters_score[0][0]
        del clusters_score[0]
        topic = topics[label]
        subwords = topic[-1]
        dictionary, matrix = get_words_matrix(subwords, word2vec_model)
        clusters = cluster(matrix, dictionary, 2, 10)
        new_topics = []
        for c in clusters:
            new_topic = topic[:-1]
            new_topic.extend(list(c))
            new_topics.append(new_topic)
        new_topics.extend(topics[:label])
        new_topics.extend(topics[label + 1:])
        new_labels, new_samples_score = silhouette_samples(new_topics, word2vec_model)
        clusters_score = silhouette_clusters(new_labels, new_samples_score)
        clusters_score.sort(key=lambda x: x[1])
        topics = new_topics
        for t in topics:
            if len(t) -1 > depth:
                depth = len(t) - 1
    return topics


# 计算聚类结果中每个数据点的轮廓系数
def silhouette_samples(clusters, word2vec_model):
    labels = []
    matrix = []
    for i in range(len(clusters)):
        words = clusters[i][-1]
        _, mat = get_words_matrix(words, word2vec_model)
        for j in range(len(mat)):
            matrix.append(list(mat[j]))
            labels.append(i)
    matrix = np.array(matrix)
    labels = np.array(labels)
    samples_score = metrics.silhouette_samples(matrix, labels)
    return labels, samples_score


# 计算聚类结果中每个簇的轮廓系数
# 用簇内所有数据点的轮廓系数的均值作为这个簇的轮廓系数
def silhouette_clusters(labels, samples_score):
    clusters_score = dict()
    clusters_samples_score = dict()
    for i in range(len(labels)):
        clusters_samples_score.setdefault(labels[i], []).append(samples_score[i])
    for label, scores in clusters_samples_score.items():
        clusters_score[label] = np.array(scores).mean()
    return clusters_score.items()


def wiki_main():
    topics = hierarchical_topic_analyse_with_silhouette(
            '../data/wiki_models/200k/wiki_lda_20_model.md',
            '../data/wiki_models/wiki_word2vec_min5_model.md',
            4)
    tree = utilities.Tree(topics)
    tree.save_to_txt('topics_txt')
    tree.save('topics')


def sougou_main():
    topics = hierarchical_topic_analyse_with_silhouette(
            '../data/lda_clusters/sougou_lda_20_filter',
            '../data/models/sougou_word2vec_min5_model.md',
            True,
            3)
    tree = utilities.Tree(topics)
    tree.save_to_txt('../data/topics/sougou_lda_20_d_3_topics_txt')
    tree.save('../data/topics/sougou_lda_20_d_3_topics')


def filter_main():
    topics, useless = lda_terms_analysis('../data/models/sougou_lda_20_model.md', '../data/models/sougou_word2vec_min5_model.md')
    tp_fn = '../data/lda_clusters/sougou_lda_20_filter'
    pickle.dump(topics, open(tp_fn, 'wb'))
    utilities.Tree(topics).save_to_txt(tp_fn + '_txt')
    ul_fn = '../data/lda_clusters/sougou_lda_20_useless'
    pickle.dump(useless, open(ul_fn, 'wb'))
    utilities.Tree(useless).save_to_txt(ul_fn + '_txt')


if __name__ == '__main__':
    sougou_main()
    # filter_main()
