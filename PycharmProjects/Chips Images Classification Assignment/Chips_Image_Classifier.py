import cv2
import numpy as np

np.random.seed(1337)
import os
import tensorflow as tf

tf.set_random_seed(1337)
from sklearn.model_selection import train_test_split
import datetime
from sklearn.metrics import confusion_matrix


# generate the text
def text_create(starttime, x_test, running_time, val_loss, val_acc, tp, fp, fn, tn):
    # generate output message
    msg = '''
    ------------ %s --------------

    Testing result: 

    Testing samples : %d
    Running time : %.4f seconds
    Loss : %.4f
    Accuracy : %.4f                                                     
        Predicted condition                     True condition
                                   Condition positive      Condition negative
    Predicted condition positive   True Positive: %-4d     False Positive: %d      
    Predicted condition negative   False Negative: %-4d    True Negative: %d
    ''' % (starttime.strftime('%Y-%m-%d %H:%M:%S'),
           len(x_test), running_time, val_loss, val_acc, tp, fp, fn, tn)
    file = open('log.txt', 'a')
    file.write(msg)
    file.close()


# data augmentation
def img_rotation(img, file, data_path):
    imgtopleft = cv2.flip(img, 0)
    imgbottomright = cv2.flip(img, 1)
    imgtopright = cv2.flip(img, -1)
    # flip the acceptable images to create more faulty examples
    cv2.imwrite(data_path + 'faulty/' + file[:-4] + '_top_left.jpg', imgtopleft)
    cv2.imwrite(data_path + 'faulty/' + file[:-4] + '_bottom_right.jpg', imgbottomright)
    cv2.imwrite(data_path + 'faulty/' + file[:-4] + '_top_right.jpg', imgtopright)


# train the data and create the model
def create_model(data_path='dataset/', false_path='dataset/faulty/'):
    # start time
    starttime = datetime.datetime.now()
    # rotate acceptable images to make them faulty
    true_files = os.listdir(data_path)
    for filename in true_files:
        if filename.startswith('chips'):
            img = cv2.imread(data_path + filename)
            img_rotation(img, filename, data_path)

    # labeling
    # X is image data set, y is label set
    X = []
    y = []
    for (dirpath, dirnames, filenames) in os.walk(data_path):
        for filename in filenames:
            if filename.endswith('.jpg'):
                img = cv2.imread(os.path.join(dirpath, filename))
                for i in range(5):
                    row = np.random.randint(1, 5)
                    col = np.random.randint(1, 5)
                    cropImg = img[row:(row + 50), col:(col + 50)]
                    X.append(cropImg)
                    if dirpath.startswith('dataset/faulty'):
                        y.append(0)
                    else:
                        y.append(1)
    y = np.asarray(y)
    X = np.asarray(X)

    # split the data into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # normalize the data
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    # build the model
    early_stopping_monitor = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=0)

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation=tf.nn.relu, input_shape=(50, 50, 3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation=tf.nn.relu))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # train the training data set
    epoch = 2  # 10
    model.fit(x_train, y_train, batch_size=16, epochs=epoch, callbacks=[early_stopping_monitor])

    # generate testing result
    val_loss, val_acc = model.evaluate(x_test, y_test)
    # fail_no = round((1-val_acc)*len(x_test))
    print(val_loss, val_acc)

    # save the model
    model.save('chips.model')

    # end time
    endtime = datetime.datetime.now()
    running_time = (endtime - starttime).seconds
    print('running time: %d s' % running_time)
    # text_create(starttime, "Training",x_test,running_time,val_loss,val_acc,fail_no)


# test the model with the images in the folder of testing_dataset (This name is assumed)
def test_model(true_path='processed_dataset/true/', false_path='processed_dataset/false/'):
    # start time
    starttime = datetime.datetime.now()
    ## labeling data
    # X is image data set, y is label set
    X = []
    y = []

    files = os.listdir(true_path)
    for filename in files:
        if filename.endswith('.jpg'):
            img = cv2.imread(true_path + filename)
            # put acceptable images into X together with the "acceptable" label noted as "1"
            X.append(img)
            y.append(1)
    # put faulty images into y together with the "not acceptable" label noted as "0"
    files = os.listdir(false_path)
    for file in files:
        if file.endswith('.jpg'):
            img = cv2.imread(false_path + file)
            X.append(img)
            y.append(0)
    y = np.asarray(y)
    X = np.asarray(X)

    # normalize the data
    X = tf.keras.utils.normalize(X, axis=1)
    # generate testing result
    new_model = tf.keras.models.load_model('chips.model')
    val_loss, val_acc = new_model.evaluate(X, y)
    y_pred = new_model.predict_classes(X)
    y_pred = list(y_pred)
    y = list(y)
    tn, fn, fp, tp = confusion_matrix(y_pred, y).ravel()
    print(tn, fn, fp, tp)

    # end time
    endtime = datetime.datetime.now()
    running_time = (endtime - starttime).seconds

    text_create(starttime, X, running_time, val_loss, val_acc, tp, fp, fn, tn)


# create_model()
test_model()
