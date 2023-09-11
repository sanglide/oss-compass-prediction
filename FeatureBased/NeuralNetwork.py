import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from tqdm import tqdm
from FeatureBased.DistanceBasedModel import DistanceBasedModel


class NeuralNetwork(DistanceBasedModel):
    def __init__(self, distName="Euclidean"):
        super(NeuralNetwork, self).__init__(distName)
        self.raw = []
        self.models = Sequential(
            [
                Dense(100, activation='relu', name='l1'),
                Dense(100, activation='relu', name='l2'),
                Dense(100, activation='relu', name='l3'),
                Dense(2, activation='linear', name='l4')
            ]
        )
        self.models.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(0.01)
        )

    def fit(self, X_train, Y_train):
        distances = []
        self.raw = X_train
        pbar = tqdm(total=len(X_train))
        for i in range(len(X_train)):
            distance = []
            for j in range(len(Y_train)):
                dist = self.distanceMeasure(X_train[i], X_train[j])
                distance.append(dist)
            distances.append(distance)
            pbar.update(1)
        pbar.close()
        self.models.fit(
            np.array(distances), Y_train,
            epochs=100
        )

    def predict(self, X_test):
        distances = []
        pbar = tqdm(total=len(X_test))
        for i in range(len(X_test)):
            distance = []
            for j in range(len(self.raw)):
                dist = self.distanceMeasure(self.raw[j], X_test[i])
                distance.append(dist)
            distances.append(distance)
            pbar.update(1)
        pbar.close()
        prediction = self.models.predict(np.array(distances))
        sf_prediction = np.zeros(len(prediction))
        for i in range(len(prediction)):
            sf_prediction[i] = np.argmax(prediction[i])
        return sf_prediction
