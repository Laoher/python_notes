# from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.models import load_model
import numpy as np
import pandas as pd
import os
import json
# import jsonlines
# import demjson
import nltk
import string

# nltk.download()
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import metrics
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from collections import Counter

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 200)


def preprocessing(path):
    def json_to_df(path):
        label = []
        sentence = []
        f_new = open(path, 'r', encoding='utf-8')
        lines = f_new.readlines()
        for line in lines:
            data = eval(line)
            label.append(data["label"])
            sentence.append(str(data["sentence"]))

        dataframe = pd.DataFrame({'label': label, 'sentence': sentence})
        return dataframe

    # remove punctuation
    def remove_punc(sentence):
        table = str.maketrans("", "", string.punctuation)
        return sentence.translate(table)

    def remove_stopwords(sentence):
        stop = set(stopwords.words("english"))
        sentence = [word.lower() for word in sentence.split() if word.lower() not in stop]
        return " ".join(sentence)

    def tokenize(df):
        cw = lambda x: " ".join(nltk.word_tokenize(x))
        df['cutword'] = df['sentence'].apply(cw)
        return df

    df = json_to_df(path)
    df["sentence"] = df.sentence.map(lambda x: remove_punc(x))
    df["sentence"] = df.sentence.map(lambda x: remove_stopwords(x))
    df = tokenize(df)

    X = df.sentence
    y = df.label

    le = LabelEncoder()
    y = le.fit_transform(y).reshape(-1, 1)
    ohe = OneHotEncoder()
    y = ohe.fit_transform(y).toarray()

    return df, X, y


def check_word_counts(tok):
    ## 使用word_index属性可以看到每次词对应的编码
    ## 使用word_counts属性可以看到每个词对应的频数
    for ii, iterm in enumerate(tok.word_index.items()):
        if ii < 10:
            print(iterm)
        else:
            break
    print("===================")
    for ii, iterm in enumerate(tok.word_counts.items()):
        if ii < 10:
            print(iterm)
        else:
            break


train_df, train_x, train_y = preprocessing('train.json')
val_df, val_x, val_y = preprocessing('dev.json')
test_df, test_x, test_y = preprocessing('test.json')


## 使用Tokenizer对词组进行编码
## 当我们创建了一个Tokenizer对象后，使用该对象的fit_on_texts()函数，以空格去识别每个词,
## 可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小。


def counter_word(text):
    count = Counter()
    for i in text.values:
        for word in i.split():
            count[word] += 1
    return count


text = train_df.sentence
counter = counter_word(text)

max_words = len(counter)
print(max_words)
max_words = 18238
max_len = 50
tok = Tokenizer(max_words)  ## 使用的最大词语数为18288
tok.fit_on_texts(train_df.sentence)

# check_word_counts(tok)

train_seq = tok.texts_to_sequences(train_df.cutword)
val_seq = tok.texts_to_sequences(val_df.cutword)
test_seq = tok.texts_to_sequences(test_df.cutword)
print(train_df.sentence[0])
print(train_seq[0])
## padding
train_seq_mat = sequence.pad_sequences(train_seq, maxlen=max_len)
val_seq_mat = sequence.pad_sequences(val_seq, maxlen=max_len)
test_seq_mat = sequence.pad_sequences(test_seq, maxlen=max_len)

print(train_seq_mat.shape)
print(val_seq_mat.shape)
print(test_seq_mat.shape)

file = open("glove.840B.300d.txt", 'r')
f = open('glove.840B.300d_filtered.txt', 'w')
lines = file.readlines()
print(len(lines))
vocab_list = []
word_vector = {}
for i in lines:
    l = i.split(" ")

    if l[0] not in string.punctuation:  # and l[0] not in set(stopwords.words("english")):
        vocab_list.append(l[0])
        word_vector[l[0]] = l[1:]
        f.write(i)
f.close()

word_index = {}

embeddings_matrix = np.zeros((len(vocab_list) + 1, 300))
for i in range(len(vocab_list)):
    # print(i)
    word = vocab_list[i]  # 每个词语
    word_index[word] = i + 1  # 词语：序号
    embeddings_matrix[i + 1] = word_vector[word]  # 词向量矩阵

print(embeddings_matrix.shape)


# createmodel

def embedding_word_model(max_words, max_len):
    model = Sequential()
    ## Embedding(词汇表大小,batch大小,每个新闻的词长)
    embedding_layer = Embedding(len(vocab_list) + 1, 300, weights=[embeddings_matrix], input_length=max_len)
    model.add(embedding_layer)
    model.add(LSTM(64, dropout=0.1))
    model.add(Dense(2, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def predict_result(pre, lab):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(pre)):
        if pre[i] == 1:
            if pre[i] == lab[i]:
                TP += 1
            else:
                FP += 1
        if pre[i] == 0:
            if pre[i] == lab[i]:
                TN += 1
            else:
                FN += 1

    print("accuracy:", (TP + TN) / (TP + FP + FN + TN))
    print("precision:", TP / (TP + FP))
    print("recall:", TP / (TP + FN))
    print("F1:", (2 * TP) / (2 * TP + FP + FN))
    print(metrics.classification_report(pre, lab))


def apply_embedding_word_model():
    model = embedding_word_model(max_words, max_len)
    model.summary()
    model_fit = model.fit(train_seq_mat, train_y, batch_size=256, epochs=20,
                          validation_data=(val_seq_mat, val_y),
                          callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)]  ## 当val-loss不再提升时停止训练
                          )
    model.save('embedding_word_model.h5')
    test_pre = model.predict(test_seq_mat)
    pre = np.argmax(test_pre, axis=1)
    lab = np.argmax(test_y, axis=1)
    predict_result(pre, lab)


apply_embedding_word_model()