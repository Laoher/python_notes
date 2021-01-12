import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense


model = Sequential([
    Dense(16, input_shape=(1,),activation='relu'),
    Dense(32,activation='relu'),
    Dense(2,activation='softmax')
])

print(model.summary())