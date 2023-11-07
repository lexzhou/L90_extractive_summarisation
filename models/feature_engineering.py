# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 16:01:08 2023

@author: 17245
"""

import numpy as np
import sys
import json
import random
# from sklearn.preprocessing import StandardScaler
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from collections import Counter
from nltk.tree import Tree
from collections import defaultdict
from nltk.parse import CoreNLPParser
from nltk.parse.corenlp import CoreNLPDependencyParser
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd 
import time 

# Word2Vec model
model = api.load("word2vec-google-news-300")


CT_parser = CoreNLPParser(url='http://localhost:9000')  # Constituency Tree parser
dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')  # Dependency Tree parser

# model = Word2Vec(vector_size=300, window=5, min_count=1, workers=4)
# model.save("word2vec.model")
# model = Word2Vec.load("word2vec.model")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess(X):
    """
    X: list of list of sentences (i.e., comprising an article)
    """
    split_articles = [[s.strip() for s in x.split('.')] for i, x in enumerate(X)]
    return split_articles

def feature_engineering(X):
    """
    This function maps a batch of sentences into the set of features of our interest
    """
    
    
    # Function to extract named entities from NER_tagger's output
    def extract_named_entities(tree):
        named_entities = []
        for subtree in tree.subtrees():
            if subtree.label() != 'S':
                entity = " ".join([token for token, pos in subtree.leaves()])
                named_entities.append((entity, subtree.label()))
        return named_entities
    
    def NER_features(sentence):
        ner_tags = ["PERSON", "NORP", "FACILITY", "ORGANIZATION", "GPE", "GSP", "LOCATION","PRODUCT", "EVENT", "WORK_OF_ART", 
                    "LAW", "LANGUAGE", "DATE", "TIME","PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]    
        labels = {"NER_" + tag + '_count': 0 for tag in ner_tags}
        
        # Tokenization and POS tagging
        tokens = word_tokenize(sentence)
        pos_tags = pos_tag(tokens)
    
        # Named Entity Recognition
        named_entities = ne_chunk(pos_tags)
    
        for w, tag in extract_named_entities(named_entities):
            labels ["NER_" + tag + '_count'] += 1
    
        labels["NER_total_count"] = len(extract_named_entities(named_entities)) # Sentences with a higher #NER might be carrying more specific information about the topic, relevant for a summary.
        
        return labels
    
    def constituency_features(sentence):
        tree_string = list(CT_parser.raw_parse(sentence))
        tree = tree_string[0]
        
        features = {}
        
        """
        Here we derive two sets of features: 
        1. Structural Features 
        2. Constituent Types (including phrase, word and punctuation labels) Counts
        """    
        ### Structural Features 
        # Tree Depth
        features["CT_depth"] = tree.height()
        
        # Branching Factor
        total_nodes = sum(1 for _ in tree.subtrees())
        total_branches = sum(len(node) for node in tree.subtrees())
        branching_factor = total_branches / total_nodes
        features["CT_branching_factor"] = branching_factor
        
        # Number of Leaf Nodes
        features["CT_leaves_num"] = len(tree.leaves())
        
        # Number of Internal Nodes
        features["internal_nodes"] = total_nodes - len(tree.leaves())
        
        ### Constituent Types Counts
        with open("../stanford-corenlp-full-2018-02-27/cons_tree_labels.json","r") as file:
            labels = json.load(file)
            for lab in labels:
                features['CT' + lab + '_count'] = 0 
        
        # Loop over all subtrees and count phrase labels
        for subtree in tree.subtrees():
            label = subtree.label()
            features['CT' + label + '_count'] += 1
        return [f for f in features.values()]
        
    
    def dependency_features(sentence):
        parses = dep_parser.parse(sentence.split())
        depen_parsing_res = [[(governor, dep, dependent) for governor, dep, dependent in parse.triples()] for parse in parses]
        parse_triples = depen_parsing_res[0]
        
        with open("../stanford-corenlp-full-2018-02-27/dependency_relations_labels.json", "r") as file:
            labels = json.load(file)
            features = {'DR_' + r + '_count': 0 for r in labels}
        
        for triple in parse_triples:
            features['DR_' + triple[1] + '_count'] += 1

        return [f for f in features.values()]
    

    # Function to compute sentence vector
    def sentence_vector(sentence, model):  # model is a Word2Vec model
        words = sentence.split()
        vector_sum = np.zeros(model.vector_size)
        word_count = 0
        for word in words:
            if word in model:
                vector_sum += model[word]
                word_count += 1
        if word_count == 0:
            return np.zeros(model.vector_size)
        return vector_sum / word_count
    
    def sentence_similarity_wrt_doc(sentences):
        """
        The similarity of each sentence to the document as a whole.
        """
        # Compute sentence vectors
        sentence_vectors = [sentence_vector(sentence, model) for sentence in sentences]
        
        # Compute cosine similarity
        cosine_similarities = cosine_similarity(sentence_vectors)
        
        return cosine_similarities
    
    similarity_mat = sentence_similarity_wrt_doc(X)
    X_features = []
    
    for i, s in enumerate(X):
        # sentence length and #words
        s_length = [len(s)]
        num_words = [len(s.split())]
        
        # sentence position
        max_sent_pos = 100  
        s_p = [0] * max_sent_pos
        if i<max_sent_pos:
            s_p[i] = 1
        s_p_rev = [0] * max_sent_pos
        if ((len(X)-1)-i)<max_sent_pos:
            s_p_rev[(len(X)-1)-i] = 1 
        
        # Named Entity Recognition tag
        NER_tag = [tag for tag in NER_features(s).values()] 
        
        # constituency features
        try:  
            CF = constituency_features(s)
        except:
            CF = [0 for _ in range(77)]
            
        # dependency features
        try:
            DF = dependency_features(s)
        except:
            DF = [0 for _ in range(39)]
        
        # word-level embeddings
        words = s.split()
        word_vectors = []
        count = 0
        for word in words:
            if word in model and count<100:
                word_vectors.append(model[word])
                count +=1
                
        if len(word_vectors) < 100:  
            word_vectors += [np.zeros(model.vector_size) for _ in range(100 - len(word_vectors))]
        else:
            word_vectors = word_vectors[:100]
        word_embedding = [v for vec in word_vectors for v in vec]  # Flatten the list of vectors
        
        # sentence-level embedding
        sentence_embedding = sentence_vector(s, model)
        sentence_embedding = [v for v in sentence_embedding]
        
        # similarity score
        similarity_score = [np.mean(np.delete(similarity_mat[i], i))]
        
        # appending features
        X_features.append(s_length + NER_tag + CF + DF + sentence_embedding + similarity_score + num_words + s_p + s_p_rev + word_embedding)

    return X_features 

with open("../data/train.greedy_sent.json", 'r') as f:
    train_data = json.load(f)
    
with open("../data/validation.json", 'r') as f:
    val_data = json.load(f)
    
with open("../data/test.json", 'r') as f:
    test_data = json.load(f)

# list of lists (each list has the sentences of an article)
train_articles = [article['article'] for article in train_data]
train_highligt_decisions = [article['greedy_n_best_indices'] for article in train_data]
val_articles = [article['article'] for article in val_data]
test_articles = [article['article'] for article in test_data]

preprocessed_train_articles = preprocess(train_articles)  
preprocessed_val_articles = preprocess(val_articles)  
preprocessed_test_articles =  preprocess(test_articles)  

X_train, y_train = preprocessed_train_articles, train_highligt_decisions
X_val = preprocessed_val_articles
X_test= preprocessed_test_articles

# X_train
if True:
    dfs = []
    for i in range(0, len(X_train)):
        """
        IMPORTANT: it's considered that batch_size == #sentences
        """
        X_batch = X_train[i]  
        y_batch = y_train[i]
        
        
        """
        NOW: Do feature engineering for each article's sentences (batch) and then do forward & backprop
        """ 
        
        X_batch = np.array(feature_engineering(X_batch))

        y_batch = np.array(y_batch).reshape(-1, 1)  # Reshape to make it a column vector
        
        # Create a temporary DataFrame to hold the current batch's data
        # Assuming your data is 2D where each row is a data point
        temp_df = pd.DataFrame(np.hstack((X_batch, y_batch)), columns=[f'feature_{j}' for j in range(X_batch.shape[1])]+['target'])
    
        # Append the temporary DataFrame to df
        dfs.append(temp_df)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv('train_processed.csv', index=False)
    
# X_val
if True:
    dfs = []
    for i in range(0, len(X_val)):
        print(i)
        """
        IMPORTANT: it's considered that batch_size == #sentences
        """
        X_batch = X_val[i]  
        
        """
        NOW: Do feature engineering for each article's sentences (batch) and then do forward & backprop
        """ 
        
        X_batch = np.array(feature_engineering(X_batch))
        
        # Create a temporary DataFrame to hold the current batch's data
        temp_df = pd.DataFrame(X_batch, columns=[f'feature_{j}' for j in range(X_batch.shape[1])])
    
        # Append the temporary DataFrame to df
        dfs.append(temp_df)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv('validation_processed.csv', index=False)
    

# X_test
if True:
    dfs = []
    for i in range(0, len(X_test)):
        print(i)
        """
        IMPORTANT: it's considered that batch_size == #sentences
        """
        X_batch = X_test[i]  
        
        """
        NOW: Do feature engineering for each article's sentences (batch) and then do forward & backprop
        """ 
        
        X_batch = np.array(feature_engineering(X_batch))
        
        # Create a temporary DataFrame to hold the current batch's data
        # Assuming your data is 2D where each row is a data point
        temp_df = pd.DataFrame(X_batch, columns=[f'feature_{j}' for j in range(X_batch.shape[1])])
    
        # Append the temporary DataFrame to df
        dfs.append(temp_df)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv('test_processed.csv', index=False)
    