import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from sklearn import svm
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping


def process_embedding_word_file():
    file = open("glove.840B.300d.txt", 'r')
    f = open('glove.840B.300d_filtered.txt', 'w')
    lines = file.readlines()

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
        word = vocab_list[i]  # 每个词语
        word_index[word] = i + 1  # 词语：序号
        embeddings_matrix[i + 1] = word_vector[word]  # 词向量矩阵
    return embeddings_matrix


def preprocessing(path):
    # process json file to dataframe
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

    # remove stopwords and change to lower case
    def remove_stopwords(sentence):
        stop = set(stopwords.words("english"))
        sentence = [word.lower() for word in sentence.split() if word.lower() not in stop]
        return " ".join(sentence)

    # tokenization
    def tokenize(df):
        cw = lambda x: nltk.word_tokenize(x)
        df['cutword'] = df['sentence'].apply(cw)
        return df

    df = json_to_df(path)
    df["sentence"] = df.sentence.map(lambda x: remove_punc(x))
    df["sentence"] = df.sentence.map(lambda x: remove_stopwords(x))
    ## Tokenize
    df = tokenize(df)

    X = df.sentence
    y = df.label

    y_1d = df.label.values
    le = LabelEncoder()
    y = le.fit_transform(y).reshape(-1, 1)

    ohe = OneHotEncoder()
    y = ohe.fit_transform(y).toarray()

    return df, X, y, y_1d


def add_value_column(df):
    def vector(cutword):
        sentence = []
        for w in cutword:
            if w in word_vector.keys():
                sentence.append(word_vector[w])
            else:
                sentence.append([0] * 300)
        return sentence

    def vector_mean(vector):
        a = np.asarray(vector).astype(float)
        b = np.mean(a, axis=0)
        # print(b.shape)
        return b.tolist()

    df["value"] = df.cutword.map(lambda x: vector(x))
    df["value"] = df.value.map(lambda x: vector_mean(x))

# create models
## LSTM model
def LSTM_model(max_words, max_len):
    model = Sequential()
    ## Embedding(词汇表大小,batch大小,每个新闻的词长)
    model.add(Embedding(max_words, 32, input_length=max_len))
    model.add(LSTM(64, dropout=0.1))
    model.add(Dense(2, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    model.fit(train_seq_mat, train_y, batch_size=128, epochs=20,
              validation_data=(val_seq_mat, val_y),
              callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)]  ## stop when val-loss not decrease
              )
    # model.save('LSTM_model.h5')
    test_pre = model.predict(test_seq_mat)
    pre = np.argmax(test_pre, axis=1)
    lab = np.argmax(test_y, axis=1)
    predict_result(pre, lab, "LSTM")

## LSTM model with embedding word imported
def embedding_word_model(max_words, max_len):
    model = Sequential()
    ## Embedding word included
    embedding_layer = Embedding(max_words, 300, weights=[embeddings_matrix], input_length=max_len)
    model.add(embedding_layer)
    model.add(LSTM(64, dropout=0.1))
    model.add(Dense(2, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    model.fit(train_seq_mat, train_y, batch_size=256, epochs=20,
              validation_data=(val_seq_mat, val_y),
              callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)]  ## stop when val-loss not decrease
              )
    # model.save('embedding_word_model.h5')
    test_pre = model.predict(test_seq_mat)
    pre = np.argmax(test_pre, axis=1)
    lab = np.argmax(test_y, axis=1)
    predict_result(pre, lab, "LSTM(applying embedding word file)")


## Decision Tree
def decision_tree():
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(to_array(train_df), train_y)

    test_pre = clf.predict(to_array(test_df))
    pre = np.argmax(test_pre, axis=1)
    lab = np.argmax(test_y, axis=1)
    predict_result(pre, lab, "Decision Tree")


## SVM model
def SVM():
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(to_array(train_df), train_y_1d)
    # predict the labels on validation dataset
    pre = SVM.predict(to_array(test_df))
    lab = test_y_1d
    predict_result(pre, lab, "SVM")


def to_array(df):
    ls = []
    for i in df['value']:
        ls.append(i)
    return np.asarray(ls)


def predict_result(pre, lab, Model):
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
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = (2 * TP) / (2 * TP + FP + FN)
    print("accuracy:", accuracy)
    print("precision:", precision)
    print("recall:", recall)
    print("F1:", F1)
    if Model =="LSTM":
        f = open('result.txt', 'w')
    else:
        f = open('result.txt', 'a')
    f.write(Model + " model result on test set: \n============================\n" + "accuracy: " + str(accuracy) + "\n" +
            "precision: " + str(precision) + "\n" +
            "recall: " + str(recall) + "\n" +
            "F1: " + str(F1) + "\n")
    msg = '''
-----------------
Confusion Matrix: 

Testing samples : %d                                                   
    Predicted condition                     True condition
                               Condition positive      Condition negative
Predicted condition positive   True Positive: %d     False Positive: %d      
Predicted condition negative   False Negative: %d    True Negative: %d

''' % (len(pre), TP, FP, FN, TN)
    f.write(msg)
    f.close()

vocab_list = []
word_vector = {}
embeddings_matrix = process_embedding_word_file()
train_df, train_x, train_y, train_y_1d = preprocessing('train.json')
val_df, val_x, val_y, val_y_1d = preprocessing('dev.json')
test_df, test_x, test_y, test_y_1d = preprocessing('test.json')

# max_words, max_len
max_words = len(vocab_list) + 1
max_len = 50
tok = Tokenizer(max_words)
tok.fit_on_texts(train_x)

# check_word_counts(tok)
add_value_column(train_df)
add_value_column(val_df)
add_value_column(test_df)

## sequence of text
train_seq = tok.texts_to_sequences(train_df.cutword)
val_seq = tok.texts_to_sequences(val_df.cutword)
test_seq = tok.texts_to_sequences(test_df.cutword)

## padding
train_seq_mat = sequence.pad_sequences(train_seq, maxlen=max_len)
val_seq_mat = sequence.pad_sequences(val_seq, maxlen=max_len)
test_seq_mat = sequence.pad_sequences(test_seq, maxlen=max_len)

pipeline = Pipeline([
    ('lstm', LSTM_model(max_words, max_len)),
    ('lstm_embedding_word', embedding_word_model(max_words, max_len)),
    ('dt', decision_tree()),
    ('SVM', SVM())
])
