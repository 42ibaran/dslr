import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt

from logger import *

EPOCHS = 1000
LEARNING_RATE = 0.01

COLUMN_Y = 'Hogwarts House'
PICKLE_FILENAME = 'logreg_train.json'
PREDICTION_FILENAME = 'houses.csv'

MODE_TRAIN = 1
MODE_PREDICT = 2

class Model():
    x = None
    y = None
    thetas = None
    features = None
    config = {
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'cost_evolution': False
    }
    cost_evolution = []

    def __init__(self, df, features=None, mode=MODE_TRAIN):
        if mode == MODE_TRAIN:
            self.__init_for_train(df, features)
        elif mode == MODE_PREDICT:
            self.__init_for_predict(df)
        else:
            raise Exception("Invalid model initialization mode.")


    def __init_for_train(self, df, features):
        if features is not None:
            self.features = features
            features.append(COLUMN_Y)
            try:
                df = df[features]
            except KeyError:
                raise Exception("One or more training features you've provided are not in the dataset.")
            features.remove(COLUMN_Y)
        df = df.dropna().reset_index()
        self.x = df.select_dtypes(include=[int, float])
        self.n_features = self.x.shape[1]
        self.classes = pd.DataFrame(columns=['high_level', 'low_level'])
        self.y = df[COLUMN_Y].astype('category')
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
        df = df[self.features]
        df = df.select_dtypes(include=[int, float]).reset_index().fillna(0)
        x_norm = pd.DataFrame()
        for column in df:
            mean = self.normalization[column]['mean']
            std = self.normalization[column]['std']
            x_norm[column] = (df[column] - mean) / std
        self.x = x_norm
        self.x = pd.concat([pd.Series(np.ones(self.x.shape[0])), self.x], axis=1)

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

    def configure(self, config):
        for key in self.config.keys():
            if key in config and config[key] is not None:
                self.config[key] = config[key]

    def hypothesis(self, x):
        return 1 / (1 + np.exp(-np.dot(x, self.thetas)))

    def cost(self, guess):
        ones = np.ones(self.one_hot_encoded.shape)
        return (-1 / self.m) * np.sum((self.one_hot_encoded * np.log(guess) + (ones - self.one_hot_encoded) * np.log(ones - guess)))

    def record_cost_evolution(self, cost):
        if self.config['cost_evolution']:
            self.cost_evolution.append(cost)

    def display_cost_evolution(self):
        if not self.config['cost_evolution']:
            return
        ax = sns.lineplot(x=range(self.config['epochs']), y=self.cost_evolution)
        ax.set(xlabel='Epochs', ylabel='Cost')
        plt.show()

    def fit(self):
        for _ in range(self.config['epochs']):
            guess = self.hypothesis(self.x)
            self.record_cost_evolution(self.cost(guess))
            gradient = np.dot(self.x.T, guess - self.one_hot_encoded) / self.m
            self.thetas -= self.config['learning_rate'] * gradient
        self.display_cost_evolution()

    def predict(self):
        prediction = self.hypothesis(self.x)
        low_level_classes = np.argmax(prediction, axis=1)
        resDF = pd.DataFrame(columns=[COLUMN_Y])
        for i in low_level_classes:
            high_level_class = self.classes['high_level'][i]
            resDF = resDF.append({ COLUMN_Y : high_level_class }, ignore_index=True)
        resDF.index.name = 'Index'
        resDF.to_csv(PREDICTION_FILENAME)

    def load(self):
        with open(PICKLE_FILENAME, "r") as fi:
            try:
                obj = json.load(fi)
                self.thetas = np.array(obj['thetas'])
                self.normalization = pd.DataFrame(obj['normalization'])
                self.classes = pd.DataFrame(obj['classes'])
                self.classes = self.classes.set_index('low_level')
                self.features = obj['features']
            except:
                log.error("Binary file (%s) is invalid." % PICKLE_FILENAME)
                exit(1)

    def save(self):
        obj = {
            'thetas': self.thetas.tolist(),
            'normalization': self.normalization.to_dict(),
            'classes': self.classes.to_dict(),
            'features': self.features
        }
        with open(PICKLE_FILENAME, "w") as fi:
            json.dump(obj, fi, indent=4)
            log.info("Training result saved successfully.")

    def test(self, filename):
        try:
            datasetDF = pd.read_csv(filename, index_col=0)
            predictedDF = pd.read_csv(PREDICTION_FILENAME, index_col=0)
        except:
            log.error("One or both of files are not found or invalid csv file(s).")
            exit(1)

        datasetDF = datasetDF.reset_index()
        datasetDF = datasetDF[COLUMN_Y]
        predictedDF = predictedDF[COLUMN_Y]

        if datasetDF.shape[0] != predictedDF.shape[0]:
            log.error("Datasets are not same length.")
            exit(1)

        same = 0
        for index, house in enumerate(datasetDF):
            if house == predictedDF[index]:
                same += 1

        accuracy = 100 * same / datasetDF.shape[0]
        log.info("Accuracy: %.2f%%" % accuracy)
