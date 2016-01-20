#!/usr/bin/env python
# encoding: utf-8

import os
import re
from multiprocessing import Pool
import jieba
import jieba.posseg as pseg


# 将原始语料库中分散的文档合并起来，同一类的文档存在同一个文件里，文件中一行存一个文档。
def merge_corpus():
    in_dir = '../data/sougou_corpus'
    out_dir = '../data/m_sougou_corpus'
    #  pt = re.compile(ur'&.*?;|&nbsp')
    pt = re.compile(ur'[^\u4e00-\u9fa5\uff00-\uffef]')
    class_list = os.listdir(in_dir)
    for c in class_list:
        c_dir = os.path.join(in_dir, c)
        doc_list = os.listdir(c_dir)
        out_file = open(os.path.join(out_dir, c), 'wb')
        for doc in doc_list:
            if not doc.endswith('txt'):
                continue
            doc_string = ''
            for line in open(os.path.join(c_dir, doc)):
                try:
                    doc_string += line.decode('cp936').strip()
                except UnicodeDecodeError:
                    try:
                        doc_string += line.decode('gb18030').strip()
                    except UnicodeDecodeError:
                        print c_dir, doc
            if not doc_string:
                continue
            doc_string = pt.sub('', doc_string)
            out_file.write(doc_string.encode('utf8') + '\n')
        out_file.close()


# 将原始的搜狗语料库文档合并为一个文件，每个文档存一行，并去除无用符号
def merge_corpus_in_one_file():
    in_dir = '../data/sougou_corpus'
    out_file = '../data/sougou_corpus_all'
    out = open(out_file, 'wb')
    pt = re.compile(ur'[^\u4e00-\u9fa5\uff00-\uffef\u3000-\u303f]')
    class_list = os.listdir(in_dir)
    for c in class_list:
        c_dir = os.path.join(in_dir, c)
        doc_list = os.listdir(c_dir)
        for doc in doc_list:
            if not doc.endswith('txt'):
                continue
            doc_string = ''
            for line in open(os.path.join(c_dir, doc)):
                try:
                    text = line.decode('cp936').strip()
                    if text:
                        doc_string += text + u'。'
                except UnicodeDecodeError:
                    try:
                        text = line.decode('gb18030').strip()
                        if text:
                            doc_string += text + u'。'
                    except UnicodeDecodeError:
                        print c_dir, doc
            doc_string = pt.sub('', doc_string).strip()
            if not doc_string:
                continue
            out.write(doc_string.encode('utf8') + '\n')
    out.close()


# 提取wiki百科的中文语料，每篇文章存为一行，并去除多余的符号
def wiki_corpus(in_file, out_file):
    start_pt = re.compile(ur'<doc.*>')
    filter_pt = re.compile(ur'[^\u4e00-\u9fa5\uff00-\uffef\u3000-\u303f.]')
    out = open(out_file, 'wb')
    doc = ''
    for line in open(in_file):
        text = line.decode('utf8').strip()
        if not start_pt.match(text):
            doc += text
        else:
            doc = filter_pt.sub('', doc).strip()
            if doc:
                out.write(doc.encode('utf8') + '\n')
            doc = ''
    out.close()


# 将文件按行切分为多个文件
def partition(in_file, parts=1):
    lines = open(in_file).readlines()
    size = len(lines)
    part_size = size / parts + 1
    start = 0
    for i in range(parts):
        out_file = in_file + '_' + str(i)
        out = open(out_file, 'wb')
        part_lines = lines[start:start + part_size]
        out.writelines(part_lines)
        out.close()
        start += part_size


# 将目录下的多个文件合并为一个文件
def merge(in_dir, out_file):
    files = os.listdir(in_dir)
    out = open(out_file, 'wb')
    for f in files:
        for line in open(os.path.join(in_dir, f)):
            out.write(line)


# 对文本进行断句，输入的文本必须是Unicode编码的
def sent_tokenizer(text):
    start = 0
    i = 0
    sents = []
    punt_list = ',.!?:;~，。！？：；～'.decode('utf8')
    for s in text:
        if s in punt_list:
            sents.append(text[start: i + 1])
            start = i + 1
            i += 1
        else:
            i += 1
    if start < len(text):
        sents.append(text[start:])
    return sents


# 对文档进行分词，无词性过滤，每个句子存为一行
def tokenizer_per_sent(in_file, out_file):
    out = open(out_file, 'wb')
    count = 0
    for line in open(in_file):
        text = line.strip().decode('utf8')
        sents = sent_tokenizer(text)
        for sentence in sents:
            if len(sentence.strip()) < 2:
                continue
            # words = [w for w in jieba.cut(sentence.strip()) if len(w) > 1]
            words = jieba.cut(sentence.strip())
            if words:
                out.write('\t'.join(words).encode('utf8') + '\n')
        count += 1
        if count % 10000 == 0:
            print 'processed: ' + str(count)
    out.close()


# 多进程版本：对文档进行分词，无词性过滤，每个句子存为一行
def run_token_s(args):
    lines = args[0]
    pid = args[1]
    print 'run process ' + str(pid) + ', tasks: ' + str(len(lines))
    results = []
    count = 0
    for line in lines:
        text = line.strip().decode('utf8')
        sents = sent_tokenizer(text)
        word_list = []
        for sentence in sents:
            if len(sentence.strip()) < 2:
                continue
            word_list += jieba.cut(sentence.strip())
            if word_list:
                results.append(word_list)
        count += 1
        if count % 1000 == 0:
            print 'process ' + str(pid) + ': ' + str(count) + '/' +str(len(lines))
    print 'process ' + str(pid) + 'end!'
    return results


def paralleled_tokenizer_s(in_file, out_file, process_num=1):
    all_lines = open(in_file).readlines()
    step = int(len(all_lines) / process_num) + 1
    lines_list = []
    start = 0
    while start < len(all_lines):
        lines_list.append(all_lines[start: start + step])
        start += step
    pool = Pool(process_num)
    pids = range(len(lines_list))
    results = pool.map(run_token, zip(lines_list, pids))
    print 'writing data ...'
    out = open(out_file, 'wb')
    for r in results:
        for token_list in r:
            out.write('\t'.join([token[0] for token in token_list]).encode('utf8') + '\n')
    out.close()
    print 'end'


# 分词的多进程版本
def run_token(args):
    lines = args[0]
    pid = args[1]
    print 'run process ' + str(pid) + ', tasks: ' + str(len(lines))
    results = []
    count = 0
    for line in lines:
        text = line.strip().decode('utf8')
        sents = sent_tokenizer(text)
        word_list = []
        for sentence in sents:
            word_list += pseg.cut(sentence.strip())
        r = []
        for word, pos in word_list:
            if pos.startswith('n') or pos.startswith('a') or pos.startswith('d') or pos.startswith('v'):
                r.append((word, pos))
        r = [(word, pos) for word, pos in r if len(word) > 1]
        if r:
            results.append(r)
        count += 1
        if count % 1000 == 0:
            print 'process ' + str(pid) + ': ' + str(count) + '/' + str(len(lines))
    print 'process ' + str(pid) + ' end!'
    return results


def paralleled_tokenizer(in_file, out_file, process_num=1):
    all_lines = open(in_file).readlines()
    step = int(len(all_lines) / process_num)
    lines_list = []
    start = 0
    while start < len(all_lines):
        lines_list.append(all_lines[start: start + step])
        start += step
    pool = Pool(process_num)
    pids = range(len(lines_list))
    results = pool.map(run_token, zip(lines_list, pids))
    print 'writing data ...'
    out = open(out_file, 'wb')
    for r in results:
        for token_list in r:
            out.write('\t'.join([token[0] for token in token_list]).encode('utf8') + '\n')
    out.close()
    print 'end'


def token_files(in_dir, out_dir):
    files = os.listdir(in_dir)
    for f in files:
        in_file = os.path.join(in_dir, f)
        out_file = os.path.join(out_dir, f + '_p')
        paralleled_tokenizer(in_file, out_file, 20)


if __name__ == '__main__':
    # merge_corpus()
    # merge_corpus_in_one_file()
    # paralleled_tokenizer('../data/wikicorpus/parts/wiki_corpus_4', '../data/wikicorpus/token/wiki_corpus_4_p', 20)
    tokenizer_per_sent('../data/wikicorpus/wiki_corpus', '../data/wikicorpus/wiki_corpus_token_s')
    # paralleled_tokenizer_s('../data/wikicorpus/wiki_corpus', '../data/wikicorpus/wiki_corpus_token_s', 20)
    # wiki_corpus('../data/wikicorpus/wiki_chs', '../data/wikicorpus/wiki_corpus')
    # partition('../data/wikicorpus/wiki_corpus', 5)
    # token_files('../data/wikicorpus/parts', '../data/wikicorpus/token')
    # merge('../data/wikicorpus/token', '../data/wikicorpus/wiki_corpus_token_p')
