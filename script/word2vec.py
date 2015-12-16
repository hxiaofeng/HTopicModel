#!/usr/bin/env python
# encoding: utf-8

from gensim import models


# 训练词向量模型
def word2vec_train(corpus_file, out_model_file):
    corpus = []
    for line in open(corpus_file):
        text = line.decode('utf8').strip().split('\t')
        if text:
            corpus.append(text)
    model = models.Word2Vec(corpus, size=200, workers=10)
    model.save(out_model_file)


if __name__ == '__main__':
    word2vec_train('../data/sougou_corpus_token_all', '../data/models/sougou_word2vec_min5_model.md')
