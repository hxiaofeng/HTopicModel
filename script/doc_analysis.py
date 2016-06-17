#!/usr/bin/env python
# encoding: utf-8

import os
import pickle
import utilities
from gensim import corpora, models, similarities
from gensim import matutils
from sklearn import metrics
import numpy


def hierarchical_clustering(topics_filename, dictionary_filename, tfidf_model_filename, tfidf_matrix_filename, corpus_filename):
    tree = utilities.Tree(None)
    tree.load(topics_filename)
    topics = tree.get_topics_with_path()
    dictionary = corpora.Dictionary.load(dictionary_filename)
    tfidf = models.TfidfModel.load(tfidf_model_filename)
    corpus_tfidf = corpora.MmCorpus(tfidf_matrix_filename)
    topics_matrix = []
    for i in range(len(topics)):
        topic = topics[i]
        topic_bow = dictionary.doc2bow(topic[-1])
        topic_tfidf = tfidf[topic_bow]
        topics_matrix.append(topic_tfidf)
    corpus = []
    for line in open(corpus_filename):
        tokens = line.strip().split('\t')
        if tokens:
            corpus.append(tokens)
    index = similarities.Similarity('/tmp/index.sharp', topics_matrix, num_features=len(dictionary), num_best=1)
    sims = index[corpus_tfidf]
    topic2doc = dict()
    labels = []
    new_corpus = []
    for i in range(len(sims)):
        if sims[i]:
            topic_id = sims[i][0][0]
            labels.append(topic_id)
            new_corpus.append(corpus_tfidf[i])
            topic2doc.setdefault(topic_id, []).append(corpus[i])
        else:
            print i
    clusters = []
    for id in topic2doc:
        cluster = topics[id][:-1]
        cluster.append(topic2doc[id])
        clusters.append(cluster)
    labels = numpy.array(labels)
    return clusters, new_corpus, labels


def evaluate(corpus_fn, labels_fn):
    corpus = corpora.MmCorpus(corpus_fn)
    labels = pickle.load(open(labels_fn))
    corpus = matutils.corpus2csc(corpus, num_terms=corpus.num_terms).transpose()
    scores = []
    for i in range(16):
        score = metrics.silhouette_score(corpus, labels, metric='cosine', sample_size=5000)
        print score
        scores.append(score)
    sc = numpy.array(scores).mean()
    print 'mean', sc


def save_to_txt(out_dir, clusters):
    for cluster in clusters:
        filename = '_'.join([str(i) for i in cluster[:-1]])
        out = open(os.path.join(out_dir, filename), 'wb')
        for doc in cluster[-1]:
            line = '\t'.join(doc)
            out.write(line + '\n')
        out.close()



def wiki_main():
    clusters = hierarchical_clustering('../data/wiki_topics/topics_20_lda',
            '../data/wiki_models/200k/wiki.dict',
            '../data/wiki_models/200k/wiki_tfidf_model.md',
            '../data/wiki_models/200k/wiki_tfidf_matrix.mm',
            '../data/wikicorpus/wiki_corpus_token_200k')
    out_dir = '../data/clusters/wiki/wiki_lda_20_dp_4'
    for cluster in clusters:
        filename = '_'.join([str(i) for i in cluster[:-1]])
        out = open(os.path.join(out_dir, filename), 'wb')
        for doc in cluster[-1]:
            line = '\t'.join(doc)
            out.write(line + '\n')
        out.close()


def sougou_main():
    clusters, new_corpus, labels = hierarchical_clustering('../data/topics/sougou_lda_20_d_3_topics',
            '../data/models/sougou.dict',
            '../data/models/sougou_tfidf_model.md',
            '../data/models/sougou_tfidf_matrix.mm',
            '../data/sougou_corpus_token_p')
    out_dir = '../data/clusters/sougou_lda_20_d_3'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    clusters_out_fn = '../data/doc_clusters/sougou_lda_20_d_3'
    pickle.dump(clusters, open(clusters_out_fn, 'wb'))
    newcorpus_fn = '../data/doc_clusters/sougou_lda_20_d_3_newcorpus'
    corpora.MmCorpus.serialize(newcorpus_fn, new_corpus)
    labels_fn = '../data/doc_clusters/sougou_lad_20_d_3_labels'
    pickle.dump(labels, open(labels_fn, 'wb'))
    for cluster in clusters:
        filename = '_'.join([str(i) for i in cluster[:-1]])
        out = open(os.path.join(out_dir, filename), 'wb')
        for doc in cluster[-1]:
            line = '\t'.join(doc)
            out.write(line + '\n')
        out.close()


def eva_main():
    corpus_fn = '../data/doc_clusters/sougou_lda_20_d_3_newcorpus'
    labels_fn = '../data/doc_clusters/sougou_lad_20_d_3_labels'
    evaluate(corpus_fn, labels_fn)


if __name__ == '__main__':
    # sougou_main()
    eva_main()
