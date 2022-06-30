import glob
import pickle
import re
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import nltk
import numpy as np
import speech_recognition as sr
from django.shortcuts import render
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk import PorterStemmer
from sklearn.preprocessing import LabelEncoder

from unittest import result
import nltk
from django.shortcuts import render
import librosa
import speech_recognition as sr
import soundfile
import numpy as np

def home(request):
    return render(request, 'home.html')

def download(request):
    return render(request, 'Download.html')

def process(request):
    file=request.FILES['audioFile']
    err=False
    result=""
    proba=""
    prediction=[" "]

    #loading models
    model=load_model(r'C:\Users\rohit\PycharmProjects\djangoProject\templates\Model.hdf5')
    filename = r'C:\Users\rohit\PycharmProjects\SpeechTensor\templates\modelForPrediction.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    lb=LabelEncoder()
    y = ['anger', 'fearful', 'calm', 'happy', 'sad', 'surprised']
    lb.fit(y)
    r = sr.Recognizer()
    with sr.AudioFile(file) as source:
        try:
            audio_data = r.record(source)
            speechToText = r.recognize_google(audio_data)
            #Extracting sentiment from text
            sentence=speechToText
            sentence = sentence_cleaning(sentence)
            result = lb.inverse_transform(np.argmax(model.predict(sentence), axis=-1))[0]
            proba = np.max(model.predict(sentence))

            file.seek(0)
            feature = extract_features(file, mfcc=True, chroma=True, mel=True)
            feature = feature.reshape(1, -1)
            prediction = loaded_model.predict(feature)
            print(prediction[0])

        except Exception as e:
            print(e)
            speechToText="This audio file can not be processed!!\nPlease Check your internet connection\n" + str(e)
            err=True


    return render(request,'process.html',{"speechtotext": speechToText, "error":err,"file":file.name, "result":result,"proba":proba,"prediction":prediction[0]})

def sentence_cleaning(sentence):
    """Pre-processing sentence for prediction"""
    max_len=300
    vocabSize=11000
    stopwords = set(nltk.corpus.stopwords.words('english'))
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


# Extract features (mfcc,chroma,mel) from a sound file

def extract_features(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
    return result


# Emotions in the Ravdness dataset
emotions = {
    '01': 'netural',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
# Emotions to observe
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

