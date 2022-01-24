import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
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
import os


df = pd.read_csv('dontpatronizeme_pcl.tsv', sep='\t', header=None)
df = df.drop(columns = [0])
df.columns = ['ID', 'keyword', 'country_code', 'paragraph', 'label']

def smush_labels(label):
  if label > 1:
    return 1
  else:
    return 0

df['label'] = df['label'].apply(smush_labels)

print(df.label.value_counts())

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
    #applying function to the column

    df[new_col]= df[new_col].apply(lambda x: tokenization(x))


    stopwords = nltk.corpus.stopwords.words('english')
    def remove_stopwords(text):
        return [k for k in text if k not in stopwords]

    df[new_col]= df[new_col].apply(lambda x:remove_stopwords(x))

    print(df[new_col][0])
    
    porter_stemmer = PorterStemmer()

    if (stem):
        def stemming(text):
            text = str(text)
            return [porter_stemmer.stem(word) for word in text]

        df[new_col] = df[new_col].apply(lambda x: stemming(x))

    if (lemmatize):
        wordnet_lemmatizer = WordNetLemmatizer()
        def lemmatizer(text):
            #text = str(text)
            return [wordnet_lemmatizer.lemmatize(word) for word in text]
        df[new_col]=df[new_col].apply(lambda x:lemmatizer(x))

    print(df[new_col][0])
    
    count = 0
    for i, row in df.iterrows():
        if(len(row[new_col]) == 0):
            df.iloc[i][new_col] = [" "]

    df[new_col] = df[new_col].apply(' '.join)
    print(df[new_col][0])
    
    return df

df = df.reset_index(drop=True)
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = str(text)
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text

df['paragraph'] = df['paragraph'].apply(clean_text)
df['paragraph'] = df['paragraph'].str.replace('\d+', '')
df = preprocess(df, 'paragraph', 'cleaned_paragraph', stem = False, lemmatize = True)
df['paragraph'] = df['cleaned_paragraph']

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['paragraph'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(df['paragraph'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

Y = df['label'].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, shuffle = True)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.1))
model.add(Bidirectional(LSTM(100, dropout=0.1,recurrent_dropout=0.1,return_sequences=True)))
model.add(Bidirectional(LSTM(100, dropout=0.1,recurrent_dropout=0.1,return_sequences=True)))
model.add(Bidirectional(LSTM(100, dropout=0.1,recurrent_dropout=0.1,return_sequences=True)))
model.add(Bidirectional(LSTM(100, dropout=0.1,recurrent_dropout=0.1)))
model.add(Dense(4))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

epochs = 10
batch_size = 32

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

predictions = model.predict(X_test)
#print("predictions shape:", predictions.shape)

def labels2file(p, outf_path):
	with open(outf_path,'w') as outf:
		for pi in p:
			outf.write(','.join([str(k) for k in pi])+'\n')

# random predictions for task 1
preds_task1 = [[0] if k < 0.08 else [1] for k in predictions]
labels2file(preds_task1, os.path.join('res/', 'task1.txt'))

labels2file([[k] for k in Y_test], os.path.join('ref/', 'task1.txt'))