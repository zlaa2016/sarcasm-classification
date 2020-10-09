
# coding: utf-8



import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




#read data
import string
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import re

data = pd.read_json("news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json",lines=True)

data.info

stop_w= ""
for i in ENGLISH_STOP_WORDS:
    stop_w += str(i)
    
def clean(text):
    """
    """
    clean_text = []
    remove = string.punctuation + stop_w + "1234567890"
    
    text = re.sub("'","",text)
    text = re.sub("(\\d|\\W)+"," ", text)
    clean_text = [i for i in text if i not in remove ]
    
    return text
    
#data['headline'] = [str(l).lower() for l in data['headline']]
data['headline'] = [clean(l)for l in data['headline']]



X = np.asarray(data['headline'])
y = np.asarray(data['is_sarcastic'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)





sum(y_train[:4000])



train_percent = 0.7
train_cutoff = int(np.floor(train_percent*len(X) ) )



#vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(norm='l2',max_features = 6000)
X_vec = vectorizer.fit_transform(X)
X_vec.toarray()


embeddings_dict = {}

with open("glove.twitter.27B/glove.twitter.27B.25d.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector




from keras.preprocessing.text import text_to_word_sequence
array_length = 20 * 300

X_embedding = pd.DataFrame()

for l in X[:8000]:
    # Saving the first 20 words of the document as a sequence
    words = text_to_word_sequence(l)[0:20] 
    
    # Retrieving the vector representation of each word and appending it to the feature vector 
    feature_vector = []
    for word in words:
        try:
            feature_vector = np.append(feature_vector, 
                                       np.array(embeddings_dict[word]))
        except KeyError:
            # skip the word if not in dictionary
            pass
    # If the text has less then 20 words, fill remaining vector with zeros
    zeroes_to_add = array_length - len(feature_vector)
    feature_vector = np.append(feature_vector, 
                               np.zeros(zeroes_to_add)
                               ).reshape((1,-1))
    
    # Append the document feature vector to the feature table
    X_embedding = X_embedding.append( 
                                     pd.DataFrame(feature_vector))
    

    




len(X_embedding)

#Try some simple dimentionality reduction methods

from sklearn.svm import LinearSVC
tfidf_model = LinearSVC()
tfidf_model.fit(X_vec[0 : train_cutoff], y_train)
tfidf_prediction = tfidf_model.predict(X_vec[train_cutoff : (len(X)+1)])

from sklearn.svm import LinearSVC
embed_model = LinearSVC()
embed_model.fit(X_embedding[:7000], y_train[:7000])
embed_pred = embed_model.predict(X_embedding[7001:])


from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30, 2), random_state=1)
clf.fit(X_vec[0 : train_cutoff], y_train)
clf_pred = clf.predict(X_vec[train_cutoff : (len(X)+1)])


# compare tf-idf and glove embedding

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
results = pd.DataFrame(index = ['TF-IDF','Embedding'], 
          columns = ['Precision', 'Recall', 'F1 score', 'support','accuracy']
          )
results.loc['TF-IDF']['Precision', 'Recall', 'F1 score', 'support'] = precision_recall_fscore_support(
          y_test, 
          tfidf_prediction, 
          average = 'binary')
results.loc['TF-IDF']['accuracy'] = accuracy_score(y_test,tfidf_prediction)
results.loc['Embedding']['Precision', 'Recall', 'F1 score', 'support'] = precision_recall_fscore_support(
          y_train[4000:4999], 
          embed_pred, 
          average = 'binary')
results.loc['Embedding']['accuracy'] = accuracy_score(y_train[7000:7999],embed_pred)

print(results)


# compare dimentionality reduction methods

results = pd.DataFrame(index = ['SVM','MLP'], 
          columns = ['Precision', 'Recall', 'F1 score', 'support','accuracy']
          )
results.loc['SVM']['Precision', 'Recall', 'F1 score', 'support'] = precision_recall_fscore_support(
          y_test, 
          tfidf_prediction, 
          average = 'binary')
results.loc['SVM']['accuracy'] = accuracy_score(y_test,tfidf_prediction)

results.loc['MLP']['Precision', 'Recall', 'F1 score', 'support'] = precision_recall_fscore_support(
          y_test, 
          clf_pred, 
          average = 'binary')
results.loc['MLP']['accuracy'] = accuracy_score(y_test,clf_pred)

print(results)




#building classifiers
from sklearn.manifold import TSNE

X_embedded_2d = TSNE(n_components=2).fit_transform(X_train_vec.toarray()[:200])

plt.figure()
plt.scatter(X_embedded_2d[:,0],X_embedded_2d[:,1], c = y_train[:200])
plt.colorbar()




from sklearn.manifold import MDS
X_embedded_2d = MDS(n_components=2).fit_transform(X_train_vec.toarray()[:80])

plt.figure()
plt.scatter(X_embedded_2d[:,0],X_embedded_2d[:,1], c = y_train[:80])
plt.colorbar()


from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
X_embedded_3d = MDS(n_components=3).fit_transform(X_train_vec.toarray()[:200])

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(X_embedded_3d[:,0], X_embedded_3d[:,1],X_embedded_3d[:,2], c=y_train[:200])



#unsatisfying results from simple models above
#build RNN 


from __future__ import print_function
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer 
from keras import optimizers

vocab_size = 6000
embedding_dim = 16
max_length = 120
tokenizer = Tokenizer(num_words = vocab_size)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
X_sequences = tokenizer.texts_to_sequences(X_train)
padded = pad_sequences(X_sequences,maxlen=max_length)

test_sequences = tokenizer.texts_to_sequences(X_test)
test_padded = pad_sequences(test_sequences,maxlen=max_length)


# fix random seed for reproducibility
np.random.seed(47)
# load the dataset but only keep the top n words, zero the rest
top_words = 6000
# truncate and pad input sequences
# max_review_length = 500
# X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
#model.add(Embedding(top_words, embedding_vecor_length, input_length=200))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
#model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(20, return_sequences=False))
model.add(Dense(1, activation='relu'))
sgd = optimizers.SGD(lr=1000, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
history1=model.fit(X_vec_reshape[:train_cutoff], y_train, epochs=10, batch_size=100,validation_data=(X_vec_reshape[train_cutoff:], y_test))
print(model.summary())
# Final evaluation of the model
scores = model.evaluate(X_vec_reshape[train_cutoff:], y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))



#Evaluating Accuracy and Loss of the model
get_ipython().run_line_magic('matplotlib', 'inline')
acc=history1.history['accuracy']
val_acc=history1.history['val_accuracy']
loss=history1.history['loss']
val_loss=history1.history['val_loss']

epochs=range(len(acc)) #No. of epochs

#Plot training and validation accuracy per epoch
import matplotlib.pyplot as plt
plt.plot(epochs,acc,'r',label='Training Accuracy')
plt.plot(epochs,val_acc,'g',label='Testing Accuracy')
plt.legend()
plt.figure()

#Plot training and validation loss per epoch
plt.plot(epochs,loss,'r',label='Training Loss')
plt.plot(epochs,val_loss,'g',label='Testing Loss')
plt.legend()
plt.show()

#Building the LSTM Model
model_lstm = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_lstm.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

num_epochs = 5
model_lstm.fit(X_vec_reshape[:train_cutoff], y_train, batch_size=100,epochs=num_epochs,validation_data=(X_vec_reshape[train_cutoff:], y_test))
model_lstm.summary()
scores = model_lstm.evaluate(X_vec_reshape[train_cutoff:], y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))



#try building a CNN Model 
model_conv = tf.keras.Sequential([
    tf.keras.layers.Embedding(6000, 16, input_length=6000),
    tf.keras.layers.Conv1D(16,3,activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_conv.summary()

model_conv.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
num_epochs = 10
history2=model_conv.fit(padded, y_train, batch_size=100,epochs=num_epochs,validation_data=(test_padded, y_test))



#Evaluating Accuracy and Loss of the model
get_ipython().run_line_magic('matplotlib', 'inline')
acc=history2.history['accuracy']
val_acc=history2.history['val_accuracy']
loss=history2.history['loss']
val_loss=history2.history['val_loss']

epochs=range(len(acc)) #No. of epochs

#Plot training and validation accuracy per epoch
import matplotlib.pyplot as plt
plt.plot(epochs,acc,'r',label='Training Accuracy')
plt.plot(epochs,val_acc,'g',label='Testing Accuracy')
plt.legend()
plt.figure()

#Plot training and validation loss per epoch
plt.plot(epochs,loss,'r',label='Training Loss')
plt.plot(epochs,val_loss,'g',label='Testing Loss')
plt.legend()
plt.show()










from tensorflow.keras.preprocessing.sequence import pad_sequences
X_vec.shape
X_vec_pad = pad_sequences(X_vec,maxlen=28619)


X_vec.shape[0]
X_vec_dense = X_vec.todense()

X_vec_reshape = X_vec.toarray()[:, None,:]
X_vec_reshape.shape


X_train

#get new validation set from a difference data source - Twitter
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import API

import twitter_authentication

auth = OAuthHandler(twitter_authentication.API_KEY, twitter_authentication.API_SECRET)
api = API(auth)
auth.set_access_token(twitter_authentication.ACCESS_TOKEN, twitter_authentication.ACCESS_TOKEN_SECRET)

from tweepy import Cursor

def get_n_tweets(user,n):
    tweets = []
    for status in Cursor(api.user_timeline,id=user).items(n):
        tweets.append(status.text)
    return tweets
        
def write(filepath,user,n):
    f = open(filepath,'w')
    f.write(get_n_tweets(user,n))
    f.close()

X_test = get_n_tweets('Funnyoneliners',1000)
y_test = np.repeat(1,1000)

X_not_jokes = get_n_tweets('BBCBreaking',1000)
y_not_jokes = np.repeat(0,1000)




len(X_not_jokes)



X_test.extend(X_not_jokes)
y_test = np.concatenate((y_test, y_not_jokes))



print(len(X_test),len(y_test))


#is_joke = 1 or 0


jokes = pd.DataFrame(list(zip(X_test, y_test)), columns =['Tweets', 'is_joke']) 

from sklearn.utils import shuffle
jokes = shuffle(jokes)


def clean_tweet(text):
    """
    """
    clean_text = []
    
    for i in text:
        if i[0:2] != "RT":
            temp += i.split(" ")
    
    temp = [i.strip() for i in temp if i.strip()!=""]
    temp = [i for i in temp if( i[0] not in ["@","&","#","rt","x"] and "http" not in i)and len(i)!=1]
    remove = string.punctuation + stop_w + "1234567890" + 'RT'
    
    text = re.sub("'","",text)
    text = re.sub("(\\d|\\W)+"," ", text)
    clean_text = [i for i in temp if i not in remove ]
    
    return clean_text
    

jokes['Tweets'] = [clean(l)for l in jokes['Tweets']]


X_test = np.asarray(jokes['Tweets'])
y_test = np.asarray(jokes['is_joke'])


vocab_size = 6000
embedding_dim = 16
max_length = 120
tokenizer = Tokenizer(num_words = vocab_size)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
X_sequences = tokenizer.texts_to_sequences(X_train)
padded = pad_sequences(X_sequences,maxlen=max_length)

test_sequences_tweets = tokenizer.texts_to_sequences(X)
test_padded_tweets = pad_sequences(test_sequences_tweets,maxlen=max_length)




#vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(norm='l2',max_features = 800)
X_test_tfidf = vectorizer.fit_transform(X_test)
X_test_tfidf.toarray()










