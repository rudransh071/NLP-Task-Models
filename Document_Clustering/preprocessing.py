import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

def get_data():
    titles = open('title_list.txt').read().split('\n')
    #ensures that only the first 100 are read in
    titles = titles[:100]

    links = open('link_list_imdb.txt').read().split('\n')
    links = links[:100]

    synopses_wiki = open('synopses_list_wiki.txt').read().split('\n BREAKS HERE')
    synopses_wiki = synopses_wiki[:100]

    synopses_clean_wiki = []
    for text in synopses_wiki:
        text = BeautifulSoup(text, 'html.parser').getText()
        #strips html formatting and converts to unicode
        synopses_clean_wiki.append(text)

    synopses_wiki = synopses_clean_wiki
        
        
    genres = open('genres_list.txt').read().split('\n')
    genres = genres[:100]

    print(str(len(titles)) + ' titles')
    print(str(len(links)) + ' links')
    print(str(len(synopses_wiki)) + ' synopses')
    print(str(len(genres)) + ' genres')

    synopses_imdb = open('synopses_list_imdb.txt').read().split('\n BREAKS HERE')
    synopses_imdb = synopses_imdb[:100]

    synopses_clean_imdb = []

    for text in synopses_imdb:
        text = BeautifulSoup(text, 'html.parser').getText()
        #strips html formatting and converts to unicode
        synopses_clean_imdb.append(text)

    synopses_imdb = synopses_clean_imdb

    synopses = []

    for i in range(len(synopses_wiki)):
        item = synopses_wiki[i] + synopses_imdb[i]
        synopses.append(item)

    ranks = []

    for i in range(0,len(titles)):
        ranks.append(i)

    totalvocab_stemmed = []
    totalvocab_tokenized = []
    for i in synopses:
        allwords_stemmed = tokenize_and_stem(i)
        totalvocab_stemmed.extend(allwords_stemmed)
        
        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)

    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)

    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                     min_df=0.2, stop_words='english',
                                     use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)
    terms = tfidf_vectorizer.get_feature_names()
    return tfidf_matrix, titles, ranks, synopses, genres, vocab_frame, terms