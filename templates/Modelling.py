import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import nltk
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
from keras.models import load_model
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

def DataFormating(add):
    rawData = open(add, "r").read()
    dataParsed = rawData.replace(";", "\n").split("\n")

    textList = dataParsed[0::2]
    labelList = dataParsed[1::2]

    if(len(textList)-len(labelList)==1):
        table = pd.DataFrame({
            'Sentence': textList[:-1],
            'Emotion': labelList,

        })
        return table
    table = pd.DataFrame({
        'Sentence': textList,
        'Emotion': labelList

    })
    return table

train_data= DataFormating(r"C:\Users\rohit\PycharmProjects\djangoProject\templates\Dataset\train.txt")
test_data = DataFormating(r"C:\Users\rohit\PycharmProjects\djangoProject\templates\Dataset\test.txt")
val_data = DataFormating(r"C:\Users\rohit\PycharmProjects\djangoProject\templates\Dataset\val.txt")
print(train_data.shape)
print(test_data.shape)
print(val_data.shape)
print(train_data['Emotion'].unique())

#Processing
#1. Label Encoder
lb=LabelEncoder()

train_data['Emotion']=lb.fit_transform(train_data['Emotion'])
test_data['Emotion']=lb.fit_transform(test_data['Emotion'])
val_data['Emotion']=lb.fit_transform(val_data['Emotion'])
print(train_data.head())
print(test_data.head())
print(val_data.head())

#2. Removing stopwords and chars
nltk.download('stopwords')
stopwords= set(nltk.corpus.stopwords.words('english'))

print(stopwords)

#Tokenization
tokenizer=Tokenizer()
vocabSize = 11000
max_len=300

def text_cleaning(df,column):
  stemmer=PorterStemmer()
  #wn=nltk.WordNetLemmatizer()
  corpus=[]

  for text in df[column]:
    text=re.sub("[^a-zA-Z]"," ",text)
    text=text.lower()
    text=text.split()
    text=[stemmer.stem(word) for word in text if word not in stopwords]
    #text=[wn.lemmatize(word) for word in text if word not in stopwords]
    text=" ".join(text)
    corpus.append(text)

  one_hot_word=[one_hot(input_text=word, n=vocabSize) for word in corpus]
  print(corpus[1])
  print(one_hot_word[1])
  pad = pad_sequences(sequences=one_hot_word,maxlen=max_len,padding='pre')
  print(pad.shape)
  return pad

x_train=text_cleaning(train_data,"Sentence")
x_test=text_cleaning(test_data,"Sentence")
x_val=text_cleaning(val_data,"Sentence")

y_train = train_data["Emotion"]
y_test = test_data["Emotion"]
y_val = val_data["Emotion"]

print(y_train)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)
print(y_train)

#Modelling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional,Dropout

model = Sequential()
model.add(Embedding(input_dim=vocabSize,output_dim=150,input_length=300))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(64,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(6,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

from tensorflow.keras.callbacks import EarlyStopping
callback = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

hist = model.fit(x_train,y_train,epochs=10,batch_size=64,validation_data=(x_val,y_val), verbose=1, callbacks=[callback])

model.evaluate(x_val,y_val,verbose=1)

model.evaluate(x_test,y_test,verbose=1)

#joblib.dump(model,'NLPModel.pkl')
model.save('NLPModel.h5')

lb=LabelEncoder()
import nltk
nltk.download('stopwords')

model1=load_model(r'C:\Users\rohit\PycharmProjects\djangoProject\templates\NLPModel.h5')
y = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
lb.fit(y)
vocabSize = 11000
max_len=300

stopwords=set(nltk.corpus.stopwords.words('english'))

#model1=load_model('NLPModel.h5')
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
            "He is really sweet and caring"
            ]
for sentence in sentences:
    print(sentence)
    sentence = sentence_cleaning(sentence)
    print(model1.predict(sentence))
    print(np.argmax(model1.predict(sentence)))
    result = lb.inverse_transform(np.argmax(model1.predict(sentence), axis=-1))[0]
    proba =  np.max(model1.predict(sentence))
    print(f"{result} : {proba}\n\n")