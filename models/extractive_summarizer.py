import tqdm
import random
import json 

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
import math 
from rouge_metric import PyRouge
import tqdm
import math 


from time import sleep
import contextlib
from tqdm import tqdm

    
class RougeEvaluator:

    def __init__(self) -> None:
        self.rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=False, rouge_s=False, rouge_su=False)

    def batch_score(self, gen_summaries, reference_summaries):
        score = self.rouge.evaluate(gen_summaries, [[x] for x in reference_summaries])
        return score
    
    def score(self, gen_summary, reference_summary):
        score = self.rouge.evaluate([gen_summary], [[reference_summary]])
        return score

def evaluating_validation_set():
    evaluator = RougeEvaluator()
    
    with open("./data/validation.json", 'r') as f:
        eval_data = json.load(f)

    with open("./data/val_preds.json", 'r') as f:
        pred_data = json.load(f)

    assert len(eval_data) == len(pred_data)

    pred_sums = []
    eval_sums = []
    for eval, pred in zip(eval_data, pred_data):
        pred_sums.append(pred['summary'])
        eval_sums.append(eval['summary'])

    scores = evaluator.batch_score(pred_sums, eval_sums)
    return scores['rouge-1']["f"]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def shuffle_dataframe(df):
    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    
    # Split the DataFrame back into X and y
    X_shuffled = df_shuffled.drop('target', axis=1)
    y_shuffled = df_shuffled['target']
    
    return X_shuffled, y_shuffled

def top_k_indicator(a_preds, k):
    # Step 1: Identify the top m highest probabilities and their indices
    a_preds = [float(p) for p in a_preds]

    top_k_indices = np.argsort(a_preds)[-k:]
    
    # Step 2: Create a new list of zeros
    indicator_list = np.zeros(len(a_preds), dtype=int)
    
    # Step 3: Set the elements corresponding to the top m probabilities to 1
    indicator_list[top_k_indices] = 1
    
    return indicator_list.tolist()


def summary_extraction(prepro_articles, preds, k=3):
    i = 0
    summaries = []
    for a in prepro_articles:
        a_preds = preds[i:(i+len(a))]
        summary_index = top_k_indicator(a_preds, k)
        summary = [s for i, s in enumerate(a) if summary_index[i] == 1]
        summary = ' . '.join(summary)
        summaries.append(summary)
        i += len(a)
        
    return summaries


class ExtractiveSummarizer:  ### This is a MLP
    
    def __init__(self, hidden_dim1=64, hidden_dim2=64, output_dim=1):
        self.W1 = self.b1 = self.W2 = self.b2 = self.W3 = self.b3 = None
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim

        self.mW1 = self.vW1 = self.mW2 = self.vW2 = self.mW3 = self.vW3 = None
        self.mb1 = self.vb1 = self.mb2 = self.vb2 = self.mb3 = self.vb3 = None
        

    def forward(self, X):
        if self.W1 is None:
            self.W1 = np.random.randn(X.shape[1], self.hidden_dim1) / np.sqrt(X.shape[1])
            self.b1 = np.zeros((self.hidden_dim1, 1))
            self.W2 = np.random.randn(self.hidden_dim1, self.hidden_dim2) / np.sqrt(self.hidden_dim1)
            self.b2 = np.zeros((self.hidden_dim2, 1))
            self.W3 = np.random.randn(self.hidden_dim2, self.output_dim) / np.sqrt(self.hidden_dim2)
            self.b3 = np.zeros((self.output_dim, 1))

            self.mW1 = np.zeros_like(self.W1)
            self.vW1 = np.zeros_like(self.W1)
            self.mb1 = np.zeros_like(self.b1)
            self.vb1 = np.zeros_like(self.b1)

            self.mW2 = np.zeros_like(self.W2)
            self.vW2 = np.zeros_like(self.W2)
            self.mb2 = np.zeros_like(self.b2)
            self.vb2 = np.zeros_like(self.b2)

            self.mW3 = np.zeros_like(self.W3)
            self.vW3 = np.zeros_like(self.W3)
            self.mb3 = np.zeros_like(self.b3)
            self.vb3 = np.zeros_like(self.b3)

        self.z1 = X.dot(self.W1) + self.b1.T
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1.dot(self.W2) + self.b2.T
        self.a2 = np.maximum(0, self.z2)
        self.z3 = self.a2.dot(self.W3) + self.b3.T
        self.output = sigmoid(self.z3)
        return self.output

    def preprocess(self, X):
        """
        X: list of list of sentences (i.e., comprising an article)
        """
        
        split_articles = [[s.strip() for s in x.split('.')] for x in X]
        return split_articles
    


    def train(self, data_train, X_val, epochs=20, initial_lr=0.001, batch_size=64, beta1=0.9, beta2=0.9, epsilon=1e-8, patience=3, lambda_l2=0.001):
        """
        data_train contains X and y, where y is the last column:
            X: a dataframe containing all the features for all sentences from all articles
            y: list of yes/no decision for each sentence (as boolean)
        X_val is the validation set
        """
        learning_rate = initial_lr
        epochs_without_improvement = 0
        losses = []
        val_losses = [0]

        t = 0
        for epoch in range(epochs):
            X, y = shuffle_dataframe(data_train)

            X_shuffled, y_shuffled = X, y
            loss_counter = 0
            
            for i in range(0, len(data_train)//64):
                t += 1
                """
                IMPORTANT: it's considered that batch_size == #sentences
                """
                try:
                    X_batch = X_shuffled.iloc[i*64:i*64+64]  
                    y_batch = y_shuffled.iloc[i*64:i*64+64]

                except:
                    X_batch = X_shuffled.iloc[i*64:]  
                    y_batch = y_shuffled.iloc[i*64:]
          
                X_batch = np.array(X_batch)
                y_batch = np.array(y_batch).reshape(-1, 1)  # Reshape to make it a column vector
    
                output = self.forward(X_batch)    
                error = output - y_batch
                loss_counter += np.mean(np.square(error))
                # Weight the error by instance
                instance_weights = np.where(y_batch == 1, 17, 1) # Determine instance weights based on class labels
                weighted_error = error * instance_weights
    
                # Adjust the loss computation to use the weighted error
                mse_loss = np.mean(np.square(weighted_error))
                l2_loss = lambda_l2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)) + np.sum(np.square(self.W3)))
                total_loss = mse_loss + l2_loss

                losses.append(total_loss)

                # Adjust backpropagation to use the weighted error
                sigmoid_derivative = output * (1 - output)
                weighted_error *= sigmoid_derivative
                
                # Now continue with backpropagation as usual, but use 'weighted_error' instead of 'error'
                dW3 = (self.a2.T).dot(2 * weighted_error)
                db3 = np.sum(2 * weighted_error, axis=0, keepdims=True).T
                da2 = (2 * weighted_error).dot(self.W3.T)
                dz2 = da2 * (self.a2 > 0)
                dW2 = (self.a1.T).dot(dz2)
                db2 = np.sum(dz2, axis=0, keepdims=True).T
                da1 = dz2.dot(self.W2.T)
                dz1 = da1 * (self.a1 > 0)
                dW1 = np.dot(X_batch.T, dz1)
                db1 = np.sum(dz1, axis=0, keepdims=True).T

                # Adding the regularisation term to the gradients
                dW3 += 2 * lambda_l2 * self.W3
                dW2 += 2 * lambda_l2 * self.W2
                dW1 += 2 * lambda_l2 * self.W1

                self.mW1 = beta1 * self.mW1 + (1 - beta1) * dW1
                self.vW1 = beta2 * self.vW1 + (1 - beta2) * np.square(dW1)
                mW1_corr = self.mW1 / (1 - beta1 ** t)
                vW1_corr = self.vW1 / (1 - beta2 ** t)

                self.mW2 = beta1 * self.mW2 + (1 - beta1) * dW2
                self.vW2 = beta2 * self.vW2 + (1 - beta2) * np.square(dW2)
                mW2_corr = self.mW2 / (1 - beta1 ** t)
                vW2_corr = self.vW2 / (1 - beta2 ** t)

                self.mW3 = beta1 * self.mW3 + (1 - beta1) * dW3
                self.vW3 = beta2 * self.vW3 + (1 - beta2) * np.square(dW3)
                mW3_corr = self.mW3 / (1 - beta1 ** t)
                vW3_corr = self.vW3 / (1 - beta2 ** t)

                self.mb1 = beta1 * self.mb1 + (1 - beta1) * db1
                self.vb1 = beta2 * self.vb1 + (1 - beta2) * np.square(db1)
                mb1_corr = self.mb1 / (1 - beta1 ** t)
                vb1_corr = self.vb1 / (1 - beta2 ** t)

                self.mb2 = beta1 * self.mb2 + (1 - beta1) * db2
                self.vb2 = beta2 * self.vb2 + (1 - beta2) * np.square(db2)
                mb2_corr = self.mb2 / (1 - beta1 ** t)
                vb2_corr = self.vb2 / (1 - beta2 ** t)

                self.mb3 = beta1 * self.mb3 + (1 - beta1) * db3
                self.vb3 = beta2 * self.vb3 + (1 - beta2) * np.square(db3)
                mb3_corr = self.mb3 / (1 - beta1 ** t)
                vb3_corr = self.vb3 / (1 - beta2 ** t)

                self.W1 -= learning_rate * (mW1_corr / (np.sqrt(vW1_corr) + epsilon) + 2 * lambda_l2 * self.W1)
                self.W2 -= learning_rate * (mW2_corr / (np.sqrt(vW2_corr) + epsilon) + 2 * lambda_l2 * self.W2)
                self.W3 -= learning_rate * (mW3_corr / (np.sqrt(vW3_corr) + epsilon) + 2 * lambda_l2 * self.W3)

                self.b1 -= learning_rate * (mb1_corr / (np.sqrt(vb1_corr) + epsilon))
                self.b2 -= learning_rate * (mb2_corr / (np.sqrt(vb2_corr) + epsilon))
                self.b3 -= learning_rate * (mb3_corr / (np.sqrt(vb3_corr) + epsilon))
                


            X_val_batch = X_val
            X_val_batch = np.array(X_val_batch)
            val_preds = self.forward(X_val_batch)

            with open("./data/validation.json", 'r') as f:
                eval_data = json.load(f)

            eval_articles = [article['article'] for article in eval_data]
            preprocessed_val_articles = [[s.strip() for s in x.split('.')] for i, x in enumerate(eval_articles)]
            
            summaries = summary_extraction(preprocessed_val_articles, val_preds)
            pred_data = [{'article': article, 'summary': summary} for article, summary in zip(eval_articles, summaries)]

            with open("./data/val_preds.json", 'w') as f:
                json.dump(pred_data, f, indent=4)  # indent parameter is optional, it makes the output more readable

            val_rogue_f1 = evaluating_validation_set()
            best_val_rogue_f1 = max(val_losses)
            
            if val_rogue_f1 > best_val_rogue_f1:
                best_val_rogue_f1 = val_rogue_f1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                learning_rate *= 0.9
                
            val_losses.append(val_rogue_f1)

            if epochs_without_improvement >= patience:
                # print("Early stopping triggered.")
                break

    def predict(self, X, X_df, k=3):
        """
        X: list of list of sentences (i.e., comprising an article)
        X_df : a dataframe containing the features of each sentence
        """
        X_df = np.array(X_df)
        preds = self.forward(X_df)
        
        
        X_articles = [article['article'] for article in X]
        preprocessed_X_articles = self.preprocess(X_articles) 
        
        summaries = summary_extraction(preprocessed_X_articles, preds)
        pred_data = [{'article': article, 'summary': summary} for article, summary in zip(X_articles, summaries)]
        

        for data_point in pred_data:
            summary = data_point["summary"]
            
            yield summary