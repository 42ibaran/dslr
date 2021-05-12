import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

from logger import *

EPOCHS = 2500
LEARNING_RATE = 0.01

COLUMN_Y = 'Hogwarts House'
PICKLE_FILENAME = 'logreg_train.json'
PREDICTION_FILENAME = 'houses.csv'

MODE_TRAIN = 1
MODE_PREDICT = 2

class Model():
    df_to_train = None
    df_to_predict = None
    df_with_prediction = None

    x = None
    y = None
    thetas = None
    
    config = {
        'mode': MODE_TRAIN,
        'features': None,
        'features_select': False,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'cost_evolution': False,
        'test': False,
        'split_percent': 0.8,
    }
    cost_evolution = []

    def __init__(self, df_filename, config):
        self.configure(config)
        self.__load_dataset(df_filename)

        if self.config['mode'] == MODE_PREDICT:
            self.__load_trained_data()

        self.__select_features()

        if self.config['mode'] == MODE_TRAIN:
            self.__split_dataset()
            self.__init_for_train()
        elif self.config['mode'] == MODE_PREDICT:
            self.__init_for_predict()
        else:
            raise Exception("Invalid model initialization mode.")

    def __init_for_train(self):
        self.df_to_train = self.df_to_train.dropna().reset_index(drop=True)
        self.x = self.df_to_train.select_dtypes(include=[int, float])

        self.classes = pd.DataFrame(columns=['high_level', 'low_level'])
        self.y = self.df_to_train[COLUMN_Y].astype('category')
        self.classes.high_level = self.y.unique()
        self.y = self.df_to_train[COLUMN_Y].astype('category').cat.codes
        self.classes.low_level = self.y.unique()
        self.classes = self.classes.set_index('low_level')
    
        self.n_features = self.x.shape[1]
        self.n_classes = len(self.classes)
        self.m = self.x.shape[0]
        self.thetas = np.zeros([self.x.shape[1] + 1, self.n_classes])

        self.__one_hot_encode()
        self.__normalize_before_training()

        self.x = pd.concat([pd.Series(np.ones(self.m)), self.x], axis=1)
    
    def __init_for_predict(self):
        if self.config['mode'] == MODE_PREDICT:
            self.__load_trained_data()
        else:
            self.df_with_right_prediction = self.df_to_predict[COLUMN_Y]

        self.df_to_predict = self.df_to_predict.fillna(0)
        self.df_to_predict = self.df_to_predict.select_dtypes(include=[int, float])
        
        self.__normalize_before_prediction()
        self.x = pd.concat([pd.Series(np.ones(self.x.shape[0])), self.x], axis=1)

    def __normalize_before_training(self):
        self.normalization = pd.DataFrame(index=['std', 'mean'])
        x_norm = pd.DataFrame()
        for column in self.x:
            std = self.x[column].std()
            mean = self.x[column].mean()
            x_norm[column] = (self.x[column] - mean) / std
            self.normalization[column] = [std, mean]
        self.x = x_norm

    def __normalize_before_prediction(self):
        x_norm = pd.DataFrame()
        for column in self.df_to_predict:
            mean = self.normalization[column]['mean']
            std = self.normalization[column]['std']
            x_norm[column] = (self.df_to_predict[column] - mean) / std
        self.x = x_norm

    def __one_hot_encode(self):
        self.one_hot_encoded = np.zeros([self.m, self.n_classes])

        for i in range(0, self.n_classes):
            for j in range(0, self.m):
                if self.y[j] == i:
                    self.one_hot_encoded[j, i] = 1

    def __load_dataset(self, df_filename):
        try:
            df = pd.read_csv(df_filename, index_col=0)
        except:
            raise Exception("Data file (%s) not found or invalid csv file." % df_filename)
        
        if self.config['mode'] == MODE_TRAIN:
            self.df_to_train = df
        elif self.config['mode'] == MODE_PREDICT:
            self.df_to_predict = df
        else:
            raise Exception("Invalid model initialization mode.")

    def __select_features(self):
        if self.config['features'] is not None:
            self.config['features'].append(COLUMN_Y)
            try:
                self.df_to_train = self.df_to_train[self.config['features']] if self.df_to_train is not None else self.df_to_train
                self.df_to_predict = self.df_to_predict[self.config['features']] if self.df_to_predict is not None else self.df_to_predict
            except KeyError:
                raise Exception("Failed selecting features from dataset.")
            self.config['features'].remove(COLUMN_Y)

    def __split_dataset(self):
        if self.config['split_percent'] < 0.0 or self.config['split_percent'] > 1.0:
            raise Exception("Invalid dataset split percentage value.")

        if self.config['test']:
            df_to_train = self.df_to_train.sample(frac=self.config['split_percent'])
            self.df_to_predict = self.df_to_train.drop(df_to_train.index)
            self.df_to_train = df_to_train
            
            self.df_to_predict = self.df_to_predict.reset_index(drop=True)
            self.df_to_train = self.df_to_train.reset_index(drop=True)

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
        self.test()

    def predict(self):
        prediction = self.hypothesis(self.x)
        low_level_classes = np.argmax(prediction, axis=1)
        self.df_with_prediction = pd.DataFrame(columns=[COLUMN_Y])
        for i in low_level_classes:
            high_level_class = self.classes.loc[i]['high_level']
            self.df_with_prediction = self.df_with_prediction.append({ COLUMN_Y : high_level_class }, ignore_index=True)
        self.df_with_prediction.index.name = 'Index'
        if self.config['mode'] == MODE_PREDICT:
            self.df_with_prediction.to_csv(PREDICTION_FILENAME)
        elif self.config['mode'] != MODE_TRAIN:
            raise Exception("Invalid model initialization mode.")

    def __load_trained_data(self):
        with open(PICKLE_FILENAME, "r") as fi:
            try:
                obj = json.load(fi)
                self.thetas = np.array(obj['thetas'])
                self.normalization = pd.DataFrame(obj['normalization'])
                self.classes = pd.DataFrame(obj['classes'])
                self.config['features'] = obj['features']
            except:
                log.error("Binary file (%s) is invalid." % PICKLE_FILENAME)
                exit(1)

    def save(self):
        obj = {
            'thetas': self.thetas.tolist(),
            'normalization': self.normalization.to_dict(),
            'classes': self.classes.to_dict(),
            'features': self.config['features']
        }
        with open(PICKLE_FILENAME, "w") as fi:
            json.dump(obj, fi, indent=4)
            log.info("Training result saved successfully.")

    def test(self):
        if not self.config['test']:
            return

        self.__init_for_predict()
        self.predict()

        if self.df_with_right_prediction.shape[0] != self.df_with_prediction.shape[0]:
            log.error("Datasets are not same length.")
            exit(1)

        columns = ["Positive/Negative"]
        columns += self.classes['high_level'].tolist()
        accuracy_df = pd.DataFrame(columns=columns)
        accuracy_df["Positive/Negative"] = ["True Positive", "True Negative", "False Positive", "False Negative"]
        accuracy_df = accuracy_df.fillna(0)
        accuracy_df.set_index("Positive/Negative", inplace=True)

        self.df_with_prediction = self.df_with_prediction[COLUMN_Y]

        for index, house in enumerate(self.df_with_right_prediction):
            if self.df_with_prediction[index] == house:
                accuracy_df[house]["True Positive"] += 1
                accuracy_df.loc["True Negative", accuracy_df.columns != house] += 1
            else:
                accuracy_df[self.df_with_prediction[index]]["False Positive"] += 1
                accuracy_df[house]["False Negative"] += 1
        print(accuracy_df)
        trues = accuracy_df.loc[["True Positive", "True Negative"]].sum().sum()
        all = accuracy_df.sum().sum()
        accuracy = 100 * trues / all
        log.info("Accuracy: %.2f%%" % accuracy)
