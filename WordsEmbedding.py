from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from PreprocessData import *
import numpy as np
import nltk
import fasttext as ft
import requests
from gensim.models import KeyedVectors


def separate(string):
    tokenized = nltk.word_tokenize(string)
    #tk = requests.post('http://192.168.23.86:2673/coreml/tokenizer', data = {'text': string, 'body': 'APPLICATION_FORM_URLENCODED'})
    #print(tk.text)
    #print(type(tk))
    return tokenized#tk.text.split(" ")

def tf_idf():
    vectorizer = TfidfVectorizer(analyzer=separate)
    vectorizer = vectorizer.fit(train_data)
    # #print(vectorizer)
    #print(dict(zip(vectorizer.get_feature_names(), idf)))
    #
    train_embed = vectorizer.transform(train_data)
    test_embed = vectorizer.transform(test_data)
    # val_embed = vectorizer.transform(val_data)

    train_embed = train_embed.todense()
    train_embed = np.array(train_embed)
    #print(train_embed.shape)
    # print(train_embed)

    test_embed = test_embed.todense()
    test_embed = np.array(test_embed)
    #print(test_embed.shape)

    # val_embed = val_embed.todense()
    # val_embed = np.array(val_embed)
    #print(val_embed.shape)

    # train_embed = np.load("data/train_embed.txt")
    # test_embed = np.load("data/test_embed.txt")
    # val_embed = np.load("data/val_embed.txt")
    #print(train_embed)
    return train_embed, train_labels, test_embed, test_labels, vectorizer

def bag_of_word():
    vectorizer = CountVectorizer(analyzer=separate)
    vectorizer = vectorizer.fit(train_data)

    train_embed = vectorizer.transform(train_data)
    test_embed = vectorizer.transform(test_data)
    # val_embed = vectorizer.transform(val_data)

    train_embed = train_embed.todense()
    train_embed = np.array(train_embed)
    #print(train_embed.shape)
    # print(train_embed)

    test_embed = test_embed.todense()
    test_embed = np.array(test_embed)
    #print(test_embed.shape)

    # val_embed = val_embed.todense()
    # val_embed = np.array(val_embed)
    #print(val_embed.shape)

    # train_embed = np.load("data/train_embed.txt")
    # test_embed = np.load("data/test_embed.txt")
    # val_embed = np.load("data/val_embed.txt")
    # print(train_embed)

    return train_embed, train_labels, test_embed, test_labels

def n_gram():
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(3,3))
    vectorizer = vectorizer.fit(train_data)

    train_embed = vectorizer.transform(train_data)
    test_embed = vectorizer.transform(test_data)
    # val_embed = vectorizer.transform(val_data)

    train_embed = train_embed.todense()
    train_embed = np.array(train_embed)
    #print(train_embed.shape)
    # print(train_embed)

    test_embed = test_embed.todense()
    test_embed = np.array(test_embed)
    #print(test_embed.shape)

    # val_embed = val_embed.todense()
    # val_embed = np.array(val_embed)
    #print(val_embed.shape)

    # train_embed = np.load("data/train_embed.txt")
    # test_embed = np.load("data/test_embed.txt")
    # val_embed = np.load("data/val_embed.txt")
    # print(train_embed)

    return train_embed, train_labels, test_embed, test_labels

def fasttext(maxwords = 50):
    dimens = 100
    #ftmodel = ft.supervised('data/trainprocess.txt', 'model/train', label_prefix='__label__')
    #ftmodel = ft.load_model('model/model_sentiment.bin', encoding = 'utf-8', label_prefix='__label__')
    ftmodel = ft.skipgram('data/trainprocess.txt', 'skip_gram', dim = dimens)
    # print(len(ftmodel['langgg']))
    # print(ftmodel.words)
    train_embed = []
    test_embed = []
    count = 0
    for text in train_data:
        tokens = separate(text)
        embed = []
        for j in range(len(tokens)):
            token = tokens[j]
            if (j >= maxwords):
                count = count + 1
                break
            vec = ftmodel[token]
            for i in range(dimens):
                embed.append(vec[i])
        if (len(tokens) < maxwords):
            for j in range(len(tokens),maxwords,1):
                for i in range(dimens):
                    embed.append(0)
        train_embed.append(embed)
    print(count)
    for text in test_data:
        tokens = separate(text)
        embed = []
        for j in range(len(tokens)):
            token = tokens[j]
            if (j >= maxwords):
                count = count + 1
                break
            vec = ftmodel[token]
            for i in range(dimens):
                embed.append(vec[i])
        if (len(tokens) < maxwords):
            for j in range(len(tokens), maxwords,1):
                for i in range(dimens):
                    embed.append(0)
        test_embed.append(embed)
    print(count)
    train_embed = np.array(train_embed)
    test_embed  = np.array(test_embed)
    print(train_embed.shape)
    print(test_embed.shape)
    return train_embed, train_labels, test_embed, test_labels

def fasttext_lstm():
    dimens = 100
    #ftmodel = ft.supervised('data/trainprocess.txt', 'model/train', label_prefix='__label__')
    #ftmodel = ft.load_model('model/model_sentiment.bin', encoding = 'utf-8', label_prefix='__label__')
    ftmodel = ft.skipgram('data/trainprocess.txt', 'skip_gram', dim = dimens)
    # print(len(ftmodel['langgg']))
    # print(ftmodel.words)
    train_embed = []
    test_embed = []

    for text in train_data:
        tokens = separate(text)
        embed = []
        for token in tokens:
            vec = ftmodel[token]
            embed.append(vec)
        train_embed.append(embed)
        #print(embed)

    for text in test_data:
        tokens = separate(text)
        embed = []
        for token in tokens:
            vec = ftmodel[token]
            embed.append(vec)
        test_embed.append(embed)

    train_embed = np.array(train_embed)
    test_embed  = np.array(test_embed)
    return train_embed, train_labels, test_embed, test_labels

def fasttext_tfidf():
    dimens = 150
    vectorizer = TfidfVectorizer(analyzer=separate)
    vectorizer = vectorizer.fit(train_data)
    train_tfidf = vectorizer.transform(train_data)
    test_tfidf  = vectorizer.transform(test_data)
    vocab = vectorizer.vocabulary_
    print(type(vocab))
    print(test_tfidf.shape)
    ftmodel = ft.skipgram('data/trainprocess.txt', 'skip_gram', dim=dimens)
    # print(ftmodel.words)
    train_embed = []
    test_embed = []

    for j in range(len(train_data)):
        text = train_data[j]
        tokens = separate(text)
        embed = []
        for i in range(dimens):
            embed.append(0)
        for token in tokens:
            vec = ftmodel[token]
            multi = 1
            if (token in vocab.keys()):
                multi = train_tfidf[j, vocab[token]]
            for i in range(dimens):
                embed[i] += vec[i] * multi
        for i in range(dimens):
            embed[i] = embed[i]/(len(tokens))
        train_embed.append(embed)
        #print(embed)

    for j in range(len(test_data)):
        text = test_data[j]
        tokens = separate(text)
        embed = []
        for i in range(dimens):
            embed.append(0)
        for token in tokens:
            vec = ftmodel[token]
            multi = 1
            if (token in vocab.keys()):
                multi = test_tfidf[j, vocab[token]]
            for i in range(dimens):
                embed[i] += vec[i] * multi
        for i in range(dimens):
            embed[i] = embed[i] / (len(tokens))
        test_embed.append(embed)

    train_embed = np.array(train_embed)
    test_embed = np.array(test_embed)
    print(train_embed.shape)
    print(test_embed.shape)
    return train_embed, train_labels, test_embed, test_labels


def fasttext_pretrain(maxwords):
    vi_model = KeyedVectors.load_word2vec_format('model/wiki.vi.vec')
    print(type(vi_model))
    print(vi_model.word_vec("thể").shape)
    dimens = 300
    train_embed = []
    test_embed = []
    count = 0
    for text in train_data:
        tokens = separate(text)
        embed = []
        for j in range(len(tokens)):
            token = tokens[j]
            if (j >= maxwords):
                count = count + 1
                break
            vec = vi_model.word_vec(token)
            for i in range(dimens):
                embed.append(vec[i])
        if (len(tokens) < maxwords):
            for j in range(len(tokens), maxwords, 1):
                for i in range(dimens):
                    embed.append(0)
        train_embed.append(embed)
    print(count)
    for text in test_data:
        tokens = separate(text)
        embed = []
        for j in range(len(tokens)):
            token = tokens[j]
            if (j >= maxwords):
                count = count + 1
                break
                vec = vi_model.word_vec(token)
            for i in range(dimens):
                embed.append(vec[i])
        if (len(tokens) < maxwords):
            for j in range(len(tokens), maxwords, 1):
                for i in range(dimens):
                    embed.append(0)
        test_embed.append(embed)
    print(count)
    train_embed = np.array(train_embed)
    test_embed = np.array(test_embed)
    return train_embed, train_labels, test_embed, test_labels


def fasttext_cnn(maxwords = 50, dimens = 50):
    #ftmodel = ft.supervised('data/trainprocess.txt', 'model/train', label_prefix='__label__')
    #ftmodel = ft.load_model('model/model_sentiment.bin', encoding = 'utf-8', label_prefix='__label__')
    ftmodel = ft.skipgram('data/trainprocess.txt', 'skip_gram', dim = dimens)
    # print(len(ftmodel['langgg']))
    # print(ftmodel.words)
    train_embed = []
    test_embed = []
    count = 0
    for text in train_data:
        tokens = separate(text)
        embed = []
        for j in range(len(tokens)):
            token = tokens[j]
            if (j >= maxwords):
                count = count + 1
                break
            vec = ftmodel[token]
            embed.append(vec)
        if (len(tokens) < maxwords):
            for j in range(len(tokens),maxwords,1):
                embed.append([0] * dimens)
        train_embed.append(embed)
    print(count)
    for text in test_data:
        tokens = separate(text)
        embed = []
        for j in range(len(tokens)):
            token = tokens[j]
            if (j >= maxwords):
                count = count + 1
                break
            vec = ftmodel[token]
            embed.append(vec)
        if (len(tokens) < maxwords):
            for j in range(len(tokens), maxwords,1):
                embed.append([0] * dimens)
        test_embed.append(embed)
    print(count)
    train_embed = np.array(train_embed)
    test_embed  = np.array(test_embed)
    print(train_embed.shape)
    print(test_embed.shape)
    return train_embed, train_labels, test_embed, test_labels




def merge(train_data, train_labels, val_data, val_label):
    data = []
    labels = []
    for i in range(len(train_data)):
        data.append(train_data[i])
        labels.append(train_labels[i])
    for i in range(len(val_data)):
        data.append(val_data[i])
        labels.append(val_labels[i])
    return data,labels

train_data, train_labels = convert_to_array("data/train.txt")
test_data, test_labels = convert_to_array("data/test.txt")
val_data, val_labels = convert_to_array("data/val.txt")
train_data, train_labels = merge(train_data, train_labels, val_data, val_labels)
# fasttext_cnn(60)
#fasttext_tfidf()
#FastText()
#tf_idf()
