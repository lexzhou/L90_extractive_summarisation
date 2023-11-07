import argparse
import json
import tqdm
from models.extractive_summarizer import ExtractiveSummarizer

import pandas as pd
import numpy as np
import sys
import json
import random
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

#################################
# 1. Block for Feature Engineering
#################################

#################################
# 1.1. Loading necessary components and defining functions
#################################
# Word2Vec model
model = api.load("word2vec-google-news-300")
CT_parser = CoreNLPParser(url='http://localhost:9000')  # Constituency Tree parser
dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')  # Dependency Tree parser

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
        with open("./stanford-corenlp-full-2018-02-27/cons_tree_labels.json","r") as file:
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
        
        with open("./stanford-corenlp-full-2018-02-27/dependency_relations_labels.json", "r") as file:
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


#################################
# 1.2. Performing Feature Engineering
#################################
if True:  
    with open("./data/train.greedy_sent.json", 'r') as f:
        train_data = json.load(f)
        
    with open("./data/validation.json", 'r') as f:
        val_data = json.load(f)
        
    with open("./data/test.json", 'r') as f:
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
    dfs = []
    for i in range(0, len(X_train)):
        X_batch = X_train[i]  
        y_batch = y_train[i]
        X_batch = np.array(feature_engineering(X_batch))
        y_batch = np.array(y_batch).reshape(-1, 1)  # Reshape to make it a column vector
        temp_df = pd.DataFrame(np.hstack((X_batch, y_batch)), columns=[f'feature_{j}' for j in range(X_batch.shape[1])]+['target'])
        dfs.append(temp_df)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv('./data/train_processed.csv', index=False)
    
    # X_val
    dfs = []
    for i in range(0, len(X_val)):
        X_batch = X_val[i]  
        X_batch = np.array(feature_engineering(X_batch))
        temp_df = pd.DataFrame(X_batch, columns=[f'feature_{j}' for j in range(X_batch.shape[1])])
        dfs.append(temp_df)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv('./data/validation_processed.csv', index=False)

    # X_test
    dfs = []
    for i in range(0, len(X_test)):
        X_batch = X_test[i]  
        X_batch = np.array(feature_engineering(X_batch))
        temp_df = pd.DataFrame(X_batch, columns=[f'feature_{j}' for j in range(X_batch.shape[1])])
        dfs.append(temp_df)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv('./data/test_processed.csv', index=False)


####################################
# 2. Block for Training the Summariser
####################################
"""
Note that we've separated Feature Engineering and Summariser Training
so as to avoid repeating the feature engineering process, which is costly.
Below is the same setup provided by the lecturer but we read preprocessed data from CSV files rather than json.
"""

args = argparse.ArgumentParser()
args.add_argument('--train_data', type=str, default='./data/train.greedy_sent.json')
args.add_argument('--eval_data', type=str, default='./data/test.json')
args = args.parse_args()

model = ExtractiveSummarizer()

#############################
# 2.1. Center and Scale the data
#############################
train_data_df = pd.read_csv("./data/train_processed.csv")
val_data_df = pd.read_csv("./data/validation_processed.csv")

if args.eval_data == './data/train.json':
    eval_data_df = pd.read_csv("./data/train_processed.csv").iloc[:,:-1]
    
elif args.eval_data == './data/validation.json':
    eval_data_df = pd.read_csv("./data/validation_processed.csv")
    
elif args.eval_data == './data/test.json':
    eval_data_df = pd.read_csv("./data/test_processed.csv")

# Center and scale the training data
X_train = train_data_df.iloc[:,:-1]
train_mean = np.mean(X_train, axis=0)
train_std = np.std(X_train, axis=0)
X_train_scaled = (X_train - train_mean) / train_std
train_data_df_scaled = train_data_df
train_data_df_scaled.iloc[:,:-1] = X_train_scaled
train_data_df_scaled.fillna(0)

# scale the validation and test data using the training mean and std:
X_eval_scaled = (eval_data_df - train_mean) / train_std
X_val_scaled = (val_data_df - train_mean) / train_std

# Identify columns that contain NaN values in the training set and remove these columns on training, validation and test sets
nan_columns = train_data_df_scaled.columns[train_data_df_scaled.isna().any()].tolist()
train_data_df_scaled = train_data_df_scaled.drop(nan_columns, axis=1)
X_eval_scaled = X_eval_scaled.drop(nan_columns, axis=1)
X_val_scaled = X_val_scaled.drop(nan_columns, axis=1)
#############################
# 2.2. Training and Evaluating the model
#############################

model.train(train_data_df_scaled, X_val_scaled)

with open(args.eval_data, 'r') as f:
    eval_data = json.load(f)



summaries = model.predict(eval_data, X_eval_scaled)
eval_articles = [article['article'] for article in eval_data]
eval_out_data = [{'article': article, 'summary': summary} for article, summary in zip(eval_articles, summaries)]

print(json.dumps(eval_out_data, indent=4))