import pandas as pd
import numpy as np

COLUMN_Y = 'Hogwarts House'
# COLUMNS_X = ['Herbology', 'Defense Against the Dark Arts', 'Divination', 'History of Magic']

TRAIN_ITER = 1000
LEARNING_RATE = 0.01

class Model():
    x = None
    y = None
    thetas = None

    def __init__(self, df):
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

        # print(self.thetas)
        # print(self.class_matrix)
        # print(self.classes)

        self.__normalize()
        self.x = pd.concat([pd.Series(np.ones(self.m)), self.x], axis=1)
        print(self.one_hot_encoded)

        self.fit()
        # self.fit()
        # print(self.houses)
        # print(self.x.shape[0])
        # print(self.x.T)
        # print(pd.concat([self.y, self.x], axis=1))
        # self._normalize()

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
        return 1 / (1 + np.exp(np.dot(-x, self.thetas)))

    def cost(self, x, y):
        guess = self.hypothesis(x)
        return (-1 / self.m) * (y * np.log(guess) + (1 - y) * np.log(1 - guess)).sum()

    def fit(self):
        print(self.thetas)
        for _ in range(TRAIN_ITER):
            cost = self.cost(self.x, self.y)
            print(cost)
            exit(0)
            # for i in range(self.n_features):
                # self.thetas[i] -= (LEARNING_RATE / self.m) * (self.cost(self.x, self.y) - self.y).sum() * self.x[i]
        print(self.thetas)
