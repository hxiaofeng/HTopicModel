#!/usr/bin/env python
# encoding: utf-8

from gensim import corpora
from gensim import models


# 将分好词的语料库转化为文档-词频矩阵
def frequence_matrix(corpus_file, out_dictionary_file, out_matrix_file):
    corpus = []
    for line in open(corpus_file):
        tokens = line.strip().split('\t')
        if tokens:
            corpus.append(tokens)
    dictionary = corpora.Dictionary(corpus)
    matrix = [dictionary.doc2bow(doc) for doc in corpus]
    dictionary.save(out_dictionary_file)
    corpora.MmCorpus.serialize(out_matrix_file, matrix)


# 将文档-词频矩阵转化为tfidf矩阵
def tfidf_matrix(corpus_file, dictionary_file, out_model_file, out_matrix_file):
    corpus = corpora.MmCorpus(corpus_file)
    dictionary = corpora.Dictionary.load(dictionary_file)
    tfidf_model = models.TfidfModel(corpus, id2word=dictionary)
    corpus_tfidf = tfidf_model[corpus]
    tfidf_model.save(out_model_file)
    corpora.MmCorpus.serialize(out_matrix_file, corpus_tfidf)


if __name__ == '__main__':
    frequence_matrix('../data/wikicorpus/wiki_corpus_token_200k',
                     '../data/wiki_models/200k/wiki.dict',
                     '../data/wiki_models/200k/wiki_fre_matrix.mm')

    tfidf_matrix('../data/wiki_models/200k/wiki_fre_matrix.mm',
                 '../data/wiki_models/200k/wiki.dict',
                 '../data/wiki_models/200k/wiki_tfidf_model.md',
                 '../data/wiki_models/200k/wiki_tfidf_matrix.mm')
