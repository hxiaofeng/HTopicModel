#!/usr/bin/env python
# encoding: utf-8


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
        return topics.reverse()

    # 将树保存在文本文件中
    def save_to_txt(self, filename):
        topics = self.get_topics_with_path()
        out = open(filename, 'wb')
        for t in topics:
            out.write('\t'.join([str(p) for p in t[:-1]]) + '\n')
            out.write('\t'.join(t[-1]).encode('utf8') + '\n')
            out.write('\n')

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
