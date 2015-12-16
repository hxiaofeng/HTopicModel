#!/usr/bin/env python
# encoding: utf-8

import LDA


if __name__ == '__main__':
    ts = LDA.get_topics_terms('../data/models/sougou_lda_50_model.md')
    out = open('topics', 'wb')
    for t, terms in ts.items():
        out.write(str(t) + '\n')
        out.write('\t'.join(terms).encode('utf8') + '\n')
        out.write('\n')
    out.close()
