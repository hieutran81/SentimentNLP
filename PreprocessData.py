# -*- coding: utf-8 -*-
import re
from LoadData import *
import nltk


def clean(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = string.replace('"','')
    string = string.replace('…', '')
    string = string.replace('<', '')
    string = string.replace('>', '')
    string = string.replace(',', '')
    string = string.replace("'", "")
    string = string.replace("]", "")
    string = string.replace("[", "")
    string = string.replace("_", "")
    string = re.sub(r"[-...0-9!?.,<>“'""'_”;+%:=()#$\/]", " ", string)
    #string = re.sub(r"[...]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def separate(string):
    tokenized = nltk.word_tokenize(string)
    return tokenized

def convert_to_array(filename):
    texts, labels = load_data(filename)
    array = []
    for text in texts:
        #print(clean(text))
        # ls = separate(clean(text))
        array.append(clean(text))
    return array, labels

def print_data(filename):
    file_out = filename[:-4] + "process"+".txt"
    texts, labels = load_data(filename)
    with open(file_out, "w") as f:
        for i in range(len(texts)):
            string = str(labels[i])+" "
            string = string+ clean(texts[i])+"\n"
            print(string)
            f.write(string)
        texts, labels = load_data("data/val.txt")
        for i in range(len(texts)):
            string = str(labels[i])+" "
            string = string+ clean(texts[i])+"\n"
            print(string)
            f.write(string)

#print_data("data/train.txt")
# print_data("data/test.txt")
# print_data("data/val.txt")
# texts , labels = load_data("data/test.txt")
# separate(clean(texts[4]))
#print(convert_to_array("data/train.txt"))