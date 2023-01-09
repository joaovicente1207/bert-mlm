import pandas as pd
import unidecode
import random

def preprocessor(sentence):
    chars = 'abcdefghijklmnopqrstuvwxyz ' + 'çáãâéêíîóõôú' + '0123456789'
    sentence = sentence.lower()
    sentence_preprocessed = ''
    for word in sentence:
        if word in chars:
            sentence_preprocessed+=word
        else:
            sentence_preprocessed += unidecode.unidecode(word)
    return sentence_preprocessed
    
def create_corpus(filename, corpus):
    with open(r'./data/'+filename, 'w') as fp:
        for item in corpus:
            fp.write("%s\n" % item)
        print(filename + ' is Done')


if __name__=='__main__':

    chars = 'abcdefghijklmnopqrstuvwxyz ' + 'çáãâéêíîóõôú' + '0123456789'
    corpus_df = pd.read_csv('./data/tb_rotulado_cap_94.csv', sep='\t')

    corpus = corpus_df['text'].to_list()
    corpus = [preprocessor(text) for text in corpus]

    random.shuffle(corpus)

    corpus_len = len(corpus)
    corpus_train = corpus[:int(corpus_len*0.75)]
    corpus_dev = corpus[int(corpus_len*0.75):]

    create_corpus('corpus_train.txt',corpus_train)
    create_corpus('corpus_dev.txt',corpus_dev)


