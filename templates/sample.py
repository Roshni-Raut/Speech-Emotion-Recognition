

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
import numpy as np
import re

lb=LabelEncoder()
import nltk
nltk.download('stopwords')


from keras.models import load_model
model1=load_model('Model.hdf5')

y = ['anger', 'fearful', 'calm', 'happy', 'sad', 'surprised']
lb.fit(y)
vocabSize = 11000
max_len=300

stopwords=set(nltk.corpus.stopwords.words('english'))

def sentence_cleaning(sentence):
    stemmer = PorterStemmer()
    corpus = []
    text = re.sub("[^a-zA-Z]", " ", sentence)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    text = " ".join(text)
    corpus.append(text)
    one_hot_word = [one_hot(input_text=word, n=vocabSize) for word in corpus]
    pad = pad_sequences(sequences=one_hot_word,maxlen=max_len,padding='pre')
    return pad

sentences = [
            "He was speechles when he found out he was accepted to this new job",
            "This is outrageous, how can you talk like that?",
            "I feel like im all alone in this world",
            "He is really sweet and caring",
            "I have been feeling a little burden lately wasn't sure why that was"
            ]
for sentence in sentences:
    print(sentence)
    sentence = sentence_cleaning(sentence)
    print(model1.predict(sentence))
    result = lb.inverse_transform(np.argmax(model1.predict(sentence), axis=-1))[0]
    proba =  np.max(model1.predict(sentence))
    print(f"{result} : {proba}\n\n")