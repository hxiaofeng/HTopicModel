#!/usr/bin/env python
# encoding: utf-8
import os
import re


def merge_corpus_in_one_file():
    in_dir = '../data/jianwei/origin'
    out_file = '../data/jianwei/corpus'
    out = open(out_file, 'wb')
    pt = re.compile(ur'[^\u4e00-\u9fa5\uff00-\uffef\u3000-\u303f]')
    sub_dirs = os.listdir(in_dir)
    for d in sub_dirs:
        sub_dir = os.path.join(in_dir, d)
        doc_list = os.listdir(sub_dir)
        for doc in doc_list:
            if not doc.endswith('txt'):
                continue
            doc_string = ''
            for line in open(os.path.join(sub_dir, doc)):
                try:
                    text = line.decode('utf8').strip()
                    doc_string += text
                except UnicodeDecodeError:
                    print sub_dir, doc
            doc_string = pt.sub('', doc_string).strip()
            if not doc_string:
                continue
            out.write(doc_string.encode('utf8') + '\n')
    out.close()


if __name__ == '__main__':
    merge_corpus_in_one_file()
