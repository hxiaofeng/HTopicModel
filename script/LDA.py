#!/usr/bin/env python
# encoding: utf-8

from gensim import corpora
from gensim import models


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
    return topic2terms


if __name__ == '__main__':
    LDA_analysis('../data/wiki_models/200k/wiki_tfidf_matrix.mm',
                 '../data/wiki_models/200k/wiki.dict',
                 '../data/wiki_models/200k/wiki_lda_100_model.md',
                 100)
