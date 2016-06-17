#!/usr/bin/env python
# encoding: utf-8

import LDA
import terms_analysis
import doc_analysis
import utilities
import pickle
from gensim import corpora

LDA_TRAIN=False


def main():
    t_num = 4
    depth = 2

    tfidf_corpus_fn = '../data/jianwei/models/tfidf_matrix.mm'
    dict_fn = '../data/jianwei/models/corpus.dict'

    lda_model_fn = '../data/jianwei/models/lda_' + str(t_num) + '_model.md'
    word2vec_model_fn = '../data/jianwei/models/word2vec_min5_model.md'
    filter_topics_fn = '../data/jianwei/filters/lda_' + str(t_num) + '_filter'
    useless_topics_fn = '../data/jianwei/filters/lda_' + str(t_num) + '_useless'
    if not LDA_TRAIN:
        LDA.LDA_analysis(tfidf_corpus_fn, dict_fn, lda_model_fn, t_num)

        topics, useless = terms_analysis.lda_terms_analysis(lda_model_fn, word2vec_model_fn)
        pickle.dump(topics, open(filter_topics_fn, 'wb'))
        pickle.dump(useless, open(useless_topics_fn, 'wb'))

    topics = terms_analysis.hierarchical_topic_analyse_with_silhouette(filter_topics_fn, word2vec_model_fn, True, depth)
    tree = utilities.Tree(topics)
    hie_topics_fn = '../data/jianwei/topics/lda_' + str(t_num) + '_d_' + str(depth) + '_topics'
    tree.save(hie_topics_fn)

    tfidf_model_fn = '../data/jianwei/models/tfidf_model.md'
    doc_corpus_fn = '../data/jianwei/corpus_token_p'
    doc_clusters, doc_new_corpus, doc_labels = doc_analysis.hierarchical_clustering(
            hie_topics_fn,
            dict_fn,
            tfidf_model_fn,
            tfidf_corpus_fn,
            doc_corpus_fn)
    doc_clusters_fn = '../data/jianwei/doc_clusters/lda_' + str(t_num) + '_d_' + str(depth)
    pickle.dump(doc_clusters, open(doc_clusters_fn, 'wb'))
    doc_new_corpus_fn = doc_clusters_fn + '_newcorpus'
    corpora.MmCorpus.serialize(doc_new_corpus_fn, doc_new_corpus)
    doc_labels_fn = doc_clusters_fn + '_labels'
    pickle.dump(doc_labels, open(doc_labels_fn, 'wb'))

    print 'evaluation, lda topic: ' + str(t_num) +', depth: ' + str(depth)
    doc_analysis.evaluate(doc_new_corpus_fn, doc_labels_fn)


if __name__ == '__main__':
    main()
