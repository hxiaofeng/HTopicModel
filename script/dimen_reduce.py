#!/usr/bin/env python
# encoding: utf-8

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim import models
import terms_analysis
import pickle
import os
#  import tsne


def w2v_dimen_reduce(word2vec_fn):
    word2vec = models.Word2Vec.load(word2vec_fn)
    vocab = word2vec.vocab.keys()
    dictionary, matrix = terms_analysis.get_words_matrix(vocab, word2vec)
    #  t_matrix = tsne.tsne(matrix[:10000], 2, 50, 20.0)
    #  return t_matrix
    del vocab
    pca = PCA(n_components=50)
    pca_matrix = pca.fit_transform(matrix[:5000])
    del matrix
    tsne = TSNE(n_components=2)
    t_matrix = tsne.fit_transform(pca_matrix)
    return dictionary[:5000], t_matrix


def topic_dimen_reduce(words, word2vec):
    dictionary, matrix = terms_analysis.get_words_matrix(words, word2vec)
    pca = PCA(n_components=50)
    pca_matrix = pca.fit_transform(matrix)
    tsne = TSNE(n_components=2)
    t_matrix = tsne.fit_transform(pca_matrix)
    return dictionary, t_matrix


def get_LDA_topics(clusters):
    topics = dict()
    for c in clusters:
        topics.setdefault(c[0], []).extend(c[-1])
    return topics


def topics_dimen_reduce(topics_fn, word2vec_fn, out_dir):
    hie_topics = pickle.load(open(topics_fn))
    word2vec = models.Word2Vec.load(word2vec_fn)
    topics = get_LDA_topics(hie_topics)
    for t in topics:
        dictionary, matrix = topic_dimen_reduce(topics[t], word2vec)
        word2coord = dict()
        for i in range(len(dictionary)):
            word2coord[dictionary[i]] = matrix[i]
        out = open(os.path.join(out_dir, str(t)), 'wb')
        pickle.dump(word2coord, out)
        out.close()
