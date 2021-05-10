import pandas as pd
import numpy as np
import pickle as pk

from logger import *

COLUMN_Y = 'Hogwarts House'
# COLUMNS_X = ['Herbology', 'Defense Against the Dark Arts', 'Divination', 'History of Magic']

TRAIN_ITER = 1000
LEARNING_RATE = 0.01
PICKLE_FILENAME = 'logreg_train.pk'

MODE_TRAIN = 1
MODE_PREDICT = 2

class Model():
    x = None
    y = None
    thetas = None

    def __init__(self, df, mode=MODE_TRAIN):
        if mode == MODE_TRAIN:
            self.__init_for_train(df)
        elif mode == MODE_PREDICT:
            self.__init_for_predict(df)
        else:
            raise Exception("Wrong model initialization mode.")


    def __init_for_train(self, df):
        df = df.dropna().reset_index()
        self.x = df.select_dtypes(include=[int, float])
        self.n_features = self.x.shape[1]
        self.classes = pd.DataFrame(columns=['high_level', 'low_level'])
        self.y = df[COLUMN_Y].astype("category")
        self.classes.high_level = self.y.unique()
        self.y = df[COLUMN_Y].astype('category').cat.codes
        self.classes.low_level = self.y.unique()
        self.n_classes = len(self.classes)
        self.m = self.x.shape[0]
        self.thetas = np.zeros([self.x.shape[1] + 1, self.n_classes])
        self.one_hot_encoded = self.__one_hot_encode()
        self.__normalize()
        self.x = pd.concat([pd.Series(np.ones(self.m)), self.x], axis=1)
    
    def __init_for_predict(self, df):
        self.load()
        df = df.select_dtypes(include=[int, float]).reset_index().fillna(0)
        x_norm = pd.DataFrame()
        for column in df:
            mean = self.normalization[column]['mean']
            std = self.normalization[column]['std']
            x_norm[column] = (df[column] - mean) / std
        self.x = x_norm
        self.x = pd.concat([pd.Series(np.ones(self.x.shape[0])), self.x], axis=1)
        return

    def __normalize(self):
        self.normalization = pd.DataFrame(index=['std', 'mean'])
        x_norm = pd.DataFrame()
        for column in self.x:
            std = self.x[column].std()
            mean = self.x[column].mean()
            x_norm[column] = (self.x[column] - mean) / std
            self.normalization[column] = [std, mean]
        self.x = x_norm

    def __one_hot_encode(self):
        one_hot_encoded = np.zeros([self.m, self.n_classes])

        for i in range(0, self.n_classes):
            for j in range(0, self.m):
                if self.y[j] == i:
                    one_hot_encoded[j, i] = 1
        return one_hot_encoded
        # self.class_matrix = pd.DataFrame(self.class_matrix)

    # def softmax(self):
    #     self.x = np.exp(self.x) / np.sum(np.exp(self.x))

    def hypothesis(self, x):
        return 1 / (1 + np.exp(-np.dot(x, self.thetas)))

    def cost(self, x, y):
        guess = self.hypothesis(x)
        ones = np.ones(y.shape)

        return (-1 / self.m) * np.sum((y * np.log(guess) + (ones - y) * np.log(ones - guess)))
        # cost_matrix = np.array([0.0, 0.0, 0.0, 0.0])

        # for i in range(y.shape[1]):
        #     cost_matrix[i] = (-1 / self.m) * np.sum((y * np.log(guess[i]) + (ones - y) * np.log(ones[i] - guess[i])))

        # # print(cost_matrix)
        # return cost_matrix


    def fit(self):
        for _ in range(TRAIN_ITER):
            guess = self.hypothesis(self.x)
            gradient = np.dot(self.x.T, guess - self.one_hot_encoded) / self.m
            self.thetas -= LEARNING_RATE * gradient
            # cost = self.cost(self.x, self.one_hot_encoded)
        print(self.thetas)

    def predict(self):
        prediction = self.hypothesis(self.x)
        toto = np.argmax(prediction, axis=1)
        resList = []
        for i in toto:
            house = self.classes['high_level'][i]
            resList.append(house)
        resDF = pd.DataFrame(resList, columns=[COLUMN_Y])
        resDF.index.name = 'Index'
        resDF.to_csv('prediction.csv')

    def load(self):
        with open(PICKLE_FILENAME, "rb") as fi:
            try:
                obj = pk.load(fi)
                self.thetas = obj['thetas']
                self.normalization = obj['normalization']
                self.classes = obj['classes']
                self.classes = self.classes.set_index('low_level')
            except:
                log.error("Binary file (%s) is invalid." % PICKLE_FILENAME)
                exit(1)

    def save(self):
        obj = {
            'thetas': self.thetas,
            'normalization': self.normalization,
            'classes': self.classes
        }
        with open(PICKLE_FILENAME, "wb") as fi:
            pk.dump(obj, fi)
            log.info("Training result saved successfully.")
