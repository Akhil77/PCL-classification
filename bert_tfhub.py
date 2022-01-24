import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import os
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin")
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import re
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
STOPWORDS = set(stopwords.words('english'))
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import cufflinks

df = pd.read_csv('dontpatronizeme_pcl.tsv', sep='\t', header=None)
df = df.drop(columns = [0])
df.columns = ['ID', 'keyword', 'country_code', 'paragraph', 'label']


def smush_labels(label):
  if label > 1:
    return 1
  else:
    return 0

df['label'] = df['label'].apply(smush_labels)

df.label.value_counts()


def preprocess(df, col, new_col, stem = False, lemmatize = False):
    
    def remove_punctuation(text):
        text = str(text)
        return "".join([k for k in text if k not in string.punctuation])
    
    df[new_col]= df[col].apply(lambda x:remove_punctuation(x))
    df[new_col]= df[new_col].apply(lambda x: x.lower())

    def tokenization(text):
        text = str(text)
        tokens = word_tokenize(text)
        return tokens

    df[new_col]= df[new_col].apply(lambda x: tokenization(x))


    stopwords = nltk.corpus.stopwords.words('english')
    def remove_stopwords(text):
        return [k for k in text if k not in stopwords]

    df[new_col]= df[new_col].apply(lambda x:remove_stopwords(x))

    print(df[new_col][0])
    
    porter_stemmer = PorterStemmer()

    if (stem):
        def stemming(text):
            #text = str(text)
            return [porter_stemmer.stem(word) for word in text]

        df[new_col] = df[new_col].apply(lambda x: stemming(x))

    if (lemmatize):
        wordnet_lemmatizer = WordNetLemmatizer()
        def lemmatizer(text):
            #text = str(text)
            return [wordnet_lemmatizer.lemmatize(word) for word in text]
        df[new_col]=df[new_col].apply(lambda x:lemmatizer(x))

    
    count = 0
    for i, row in df.iterrows():
        if(len(row[new_col]) == 0):
            df.iloc[i][new_col] = [" "]

    df[new_col] = df[new_col].apply(' '.join)
    
    return df


bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4") #https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4 , trainable= True
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

# Neural network layers
l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)

# Use inputs and outputs to construct a final model
model = tf.keras.Model(inputs=[text_input], outputs = [l])

#model.summary()
METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
]

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=METRICS)

trdf1,tedf1 = train_test_split(df, test_size = 0.10, shuffle = True)
trdf1

# Downsampling imbalanced data
pcldf = trdf1[trdf1.label==1]
npos = len(pcldf)

training_set1 = pd.concat([pcldf,trdf1[trdf1.label==0][:int(npos*2.5)]])

batch_size = 32
history = model.fit(training_set1['paragraph'],training_set1['label'],epochs=10) #batch_size = batch_size

predictions_task1 = model.predict(tedf1.paragraph.tolist())

def labels2file(p, outf_path):
	with open(outf_path,'w') as outf:
		for pi in p:
			outf.write(','.join([str(k) for k in pi])+'\n')

# random predictions for task 1
preds_task1_bert = [[0] if k < 0.33 else [1] for k in predictions_task1]
labels2file(preds_task1_bert, os.path.join('res/', 'task1.txt'))


labels2file([[k] for k in tedf1['label']], os.path.join('ref/', 'task1.txt'))
