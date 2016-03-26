#!/usr/bin/env python
# encoding: utf-8

import pickle
from sklearn.cluster import KMeans
from sklearn import metrics


# Tree类，用来存储主题树，树的叶结点为词集
class Tree(object):
    def __init__(self, topics):
        if isinstance(topics, dict):
            self.root = topics
        elif isinstance(topics, list):
            self.root = self.gen_topic_tree(topics)
        else:
            self.root = dict()

    # 遍历主题树，返回词项集和路径
    def get_topics_with_path(self):
        topics = []
        stack = []
        stack.append([self.root])
        while stack:
            path = stack.pop()
            ancestors = path[:-1]
            node = path[-1]
            if isinstance(node, dict):
                for child in node:
                    new_path = ancestors[:]
                    new_path.append(child)
                    new_path.append(node[child])
                    stack.append(new_path)
            else:
                topics.append(path)
        topics.reverse()
        return topics

    # 将树保存在文本文件中
    def save_to_txt(self, filename):
        topics = self.get_topics_with_path()
        out = open(filename, 'wb')
        for t in topics:
            out.write('\t'.join([str(p) for p in t[:-1]]) + '\t' + str(len(t[-1])) + '\n')
            out.write('\t'.join(t[-1]).encode('utf8') + '\n')
            out.write('\n')

    # 将数据保存为文件
    def save(self, filename):
        topics = self.get_topics_with_path()
        pickle.dump(topics, open(filename, 'w'))

    # 从文件中导入数据
    def load(self, filename):
        topics = pickle.load(open(filename))
        self.root = self.gen_topic_tree(topics)

    # 利用带路径的主题列表构建主题树
    @staticmethod
    def gen_topic_tree(topics):
        tree = dict()
        for t in topics:
            root = tree
            for label in t[:-2]:
                root.setdefault(label, dict())
                root = root[label]
            assert t[-2] not in root
            root[t[-2]] = t[-1]
        return tree


# 用轮廓系数估计k-means聚类个数
def n_clusters_analysis(matrix, max_n=100):
    for i in range(2, max_n + 1):
        model = KMeans(i, n_jobs=10)
        model.fit(matrix)
        score = metrics.silhouette_score(matrix, model.labels_)
        print score


# 计算簇的凝聚度
def cohesion(matrix, center):
    distances = metrics.pairwise.pairwise_distances(matrix, center)
    return distances.mean()
