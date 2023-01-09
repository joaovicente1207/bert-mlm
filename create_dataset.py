import pandas as pd
import unidecode
import random

def create_corpus(filename, corpus):
    with open(r'./data/'+filename, 'w') as fp:
        for item in corpus:
            fp.write("%s\n" % item)
        print(filename + ' is Done')

def preprocessor(sentence, chars):
    sentence = unidecode.unidecode(sentence.lower())
    sentence_preprocessed = ''
    for word in sentence:
        if word in chars:
            sentence_preprocessed+=word
    
    return sentence_preprocessed

def corpus_processor(corpus):
    chars = 'abcdefghijklmnopqrstuvwxyz ' + 'çáãâéêíîóõôú' + '0123456789'
    corpus_preprocessed = []
    MAX_LENGHT_SENTENCE = 100

    for sentence in corpus:
        sentence_proc = preprocessor(sentence, chars)
        if len(sentence_proc) > 0:
            sentence_proc = sentence_proc[:MAX_LENGHT_SENTENCE] if len(sentence_proc) > MAX_LENGHT_SENTENCE else sentence_proc
            corpus_preprocessed.append(sentence_proc)

    return corpus_preprocessed

if __name__=='__main__':
    corpus = pd.read_csv('./data/tb_rotulado_cap_94.csv', sep='\t')
    corpus = corpus.query("label > 0")
    corpus = corpus['text'].to_list()

    corpus_preprocessed = corpus_processor(corpus)

    random.shuffle(corpus_preprocessed)

    corpus_len = len(corpus_preprocessed)
    corpus_train = corpus_preprocessed[:int(corpus_len*0.75)]
    corpus_dev = corpus_preprocessed[int(corpus_len*0.75):]

    create_corpus('corpus_train.txt',corpus_train)
    create_corpus('corpus_dev.txt',corpus_dev)

