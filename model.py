import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import time

from logger import *

pd.set_option('display.max_columns', 6)

DEFAULT_EPOCHS = 2500
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_SPLIT_PERCENT = 0.8

COLUMN_WITH_CLASSES = 'Hogwarts House'
JSON_FILENAME = 'logreg_train.json'
PREDICTION_FILENAME = 'houses.csv'

MODE_TRAIN = 1
MODE_PREDICT = 2

class Model():
    df_to_train = None
    df_to_predict = None
    df_model_classes = None

    x = None
    y = None
    thetas = None
    cost_evolution = []
    
    config = {
        'mode': MODE_TRAIN,
        'features': None,
        'features_select': False,
        'epochs': DEFAULT_EPOCHS,
        'learning_rate': DEFAULT_LEARNING_RATE,
        'cost_evolution': False,
        'test': False,
        'split_percent': DEFAULT_SPLIT_PERCENT,
        'seed': round(time.time()),
        'verbose': 0
    }

    def __init__(self, df_filename, config):
        self.__configure(config)
        self.__load_dataset(df_filename)

        if self.config['mode'] == MODE_TRAIN and self.config['features_select']:
            self.__prompt_feature_selector()

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

    def __configure(self, config):
        for key in self.config.keys():
            if key in config and config[key] is not None:
                self.config[key] = config[key]
        self.__verbose(1, "Model parameters:")
        self.__verbose(1, "Training iterations: " + str(self.config['epochs']))
        self.__verbose(1, "Learning rate: " + str(self.config['learning_rate']))
        self.__verbose(1, "Check accuracy: " + ("Yes" if self.config['test'] else "No"))
        if self.config['test']:
            self.__verbose(1, "Train/Test data ratio: " + str(round(100 * self.config['split_percent'])) + "/" + str(round(100 * (1.0 - self.config['split_percent']))))
            self.__verbose(1, "Seed: " + str(self.config['seed']))
        self.__verbose(1, "")

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

    def __prompt_feature_selector(self):
        numeric_features = self.df_to_train.select_dtypes(include=[int, float]).columns
        for i in range(len(numeric_features)):
            print(i + 1, '. ', numeric_features[i], sep="")
        
        print("Provide indexes of features you want to use for training (one by one):")
        indices = []
        while True:
            try:
                user_input = input()
            except KeyboardInterrupt:
                break
            if user_input == "":
                break
            try:
                index = int(user_input)
                if index < 1 or index > len(numeric_features):
                    log.warning("Ignoreing invalid value. Press enter to proceed to training.")
                    continue
                if index not in indices:
                    indices.append(index)
            except ValueError:
                log.warning("Ignoring invalid value. Press enter to proceed to training.")
        if len(indices) == 0:
            self.config['features'] = list(numeric_features)
        else:
            indices = [ i - 1 for i in indices ]
            try:
                self.config['features'] = list(numeric_features[indices])
            except IndexError:
                raise ValueError("Somehow you f***ed it up.")

    def __load_trained_data(self):
        try:
            with open(JSON_FILENAME, "r") as fi:
                obj = json.load(fi)
                self.thetas = np.array(obj['thetas'])
                self.normalization = pd.DataFrame(obj['normalization'])
                self.classes = { int(k): v for k, v in obj['classes'].items() }
                self.config['features'] = obj['features']
        except:
            raise Exception("File with training data (%s) is not found or invalid." % JSON_FILENAME)

    def __save(self):
        obj = {
            'thetas': self.thetas.tolist(),
            'normalization': self.normalization.to_dict(),
            'classes': self.classes,
            'features': self.config['features']
        }
        try:
            with open(JSON_FILENAME, "w") as fi:
                json.dump(obj, fi, indent=4)
                log.info("Training result saved successfully.")
        except:
            raise Exception("Failed saving training result.")

    def __select_features(self):
        if self.config['features'] is not None:
            self.config['features'].append(COLUMN_WITH_CLASSES)
            try:
                self.df_to_train = self.df_to_train[self.config['features']] if self.df_to_train is not None else self.df_to_train
                self.df_to_predict = self.df_to_predict[self.config['features']] if self.df_to_predict is not None else self.df_to_predict
            except KeyError:
                raise Exception("Failed selecting features from dataset.")
            self.config['features'].remove(COLUMN_WITH_CLASSES)
        self.__verbose(1, "Features for training:")
        self.__verbose(1, ", ".join(self.config['features']))
        self.__verbose(1, "")

    def __split_dataset(self):
        if self.config['split_percent'] < 0.0 or self.config['split_percent'] > 1.0:
            raise Exception("Invalid dataset split percentage value.")

        if self.config['test']:
            df_to_train = self.df_to_train.sample(frac=self.config['split_percent'], random_state=self.config['seed'])
            self.df_to_predict = self.df_to_train.drop(df_to_train.index)
            self.df_to_train = df_to_train
            
            self.df_to_predict = self.df_to_predict.reset_index(drop=True)
            self.df_to_train = self.df_to_train.reset_index(drop=True)

    def __init_for_train(self):
        self.df_to_train = self.df_to_train.dropna().reset_index(drop=True)
        self.x = self.df_to_train.select_dtypes(include=[int, float])

        self.classes = {}
        self.y = self.df_to_train[COLUMN_WITH_CLASSES].astype('category')
        classes_high_level = self.y.unique()
        classes_low_level = self.y.cat.codes.unique()
        self.classes = { int(classes_low_level[i]): classes_high_level[i] for i in range(len(classes_low_level)) }
        self.y = self.y.cat.codes

        self.__verbose(2, "Classes encoding:")
        self.__verbose(2, self.classes)
        self.__verbose(2, "")
    
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
            self.df_correct_classes = self.df_to_predict[COLUMN_WITH_CLASSES]

        self.df_to_predict = self.df_to_predict.fillna(0)
        self.df_to_predict = self.df_to_predict.select_dtypes(include=[int, float])
        
        self.__normalize_before_prediction()
        self.x = pd.concat([pd.Series(np.ones(self.x.shape[0])), self.x], axis=1)

    def __normalize_before_training(self):
        self.__verbose(1, "Normalizing dataset...")
        self.normalization = pd.DataFrame(index=['std', 'mean'])
        x_norm = pd.DataFrame()
        for column in self.x:
            std = self.x[column].std()
            mean = self.x[column].mean()
            x_norm[column] = (self.x[column] - mean) / std
            self.normalization[column] = [std, mean]
        self.x = x_norm
        self.__verbose(2, "Normalization table:")
        self.__verbose(2, self.normalization)
        self.__verbose(2, "")

    def __normalize_before_prediction(self):
        x_norm = pd.DataFrame()
        for column in self.normalization:
            mean = self.normalization[column]['mean']
            std = self.normalization[column]['std']
            x_norm[column] = (self.df_to_predict[column] - mean) / std
        self.x = x_norm

    def __one_hot_encode(self):
        self.__verbose(1, "One-hot-encoding...")
        self.one_hot_encoded = np.zeros([self.m, self.n_classes])

        for i in range(0, self.n_classes):
            for j in range(0, self.m):
                if self.y[j] == i:
                    self.one_hot_encoded[j, i] = 1
        self.__verbose(2, "One-hot-encoded table:")
        self.__verbose(2, self.one_hot_encoded)
        self.__verbose(2, "")

    def __hypothesis(self, x):
        return 1 / (1 + np.exp(-np.dot(x, self.thetas)))

    def __cost(self, guess):
        ones = np.ones(self.one_hot_encoded.shape)
        return (-1 / self.m) * np.sum((self.one_hot_encoded * np.log(guess) + (ones - self.one_hot_encoded) * np.log(ones - guess)))

    def __record_cost_evolution(self, cost):
        if self.config['cost_evolution']:
            self.cost_evolution.append(cost)

    def __display_cost_evolution(self):
        if not self.config['cost_evolution']:
            return
        ax = sns.lineplot(x=range(self.config['epochs']), y=self.cost_evolution)
        ax.set(xlabel='Epochs', ylabel='Cost')
        plt.show()

    def __evaluate(self):
        if not self.config['test']:
            return
        self.__verbose(1, "Evaluating model...")

        self.__init_for_predict()
        self.predict(save=False)

        if self.df_correct_classes.shape[0] != self.df_model_classes.shape[0]:
            raise Exception("Datasets are not same length.")

        columns = ["Metrics"]
        columns += self.classes.values()
        df_acc = pd.DataFrame(columns=columns)
        df_acc["Metrics"] = ["TP", "TN", "FP", "FN",
                             "Accuracy", "Precision"]
        df_acc = df_acc.fillna(0)
        df_acc.set_index("Metrics", inplace=True)

        self.df_model_classes = self.df_model_classes[COLUMN_WITH_CLASSES]

        for index, correct_class in enumerate(self.df_correct_classes):
            if self.df_model_classes[index] == correct_class:
                df_acc[correct_class]["TP"] += 1
                df_acc.loc["TN", df_acc.columns != correct_class] += 1
            else:
                df_acc[self.df_model_classes[index]]["FP"] += 1
                df_acc[correct_class]["FN"] += 1

        df_acc['Total'] = df_acc.sum(axis=1)
        df_acc.loc["Accuracy"] = 100 * (df_acc.loc["TP"] + df_acc.loc["TN"]) / (df_acc.loc["TP"] + df_acc.loc["TN"] + df_acc.loc["FP"] + df_acc.loc["FN"])
        df_acc.loc["Precision"] = 100 * df_acc.loc["TP"] / (df_acc.loc["TP"] + df_acc.loc["FP"])
        df_acc = df_acc.round(decimals=2)

        print(df_acc, end="\n\n")

    def __verbose(self, level, msg, end="\n"):
        if self.config['verbose'] >= level:
            print(msg, end=end)

    def __print_progress_bar(self, iteration, total):
        if iteration % 10 != 0:
            return
        length = 30
        percent = "%.1f" % (100 * iteration / total)
        filled_length = int(length * iteration // total)
        bar = '█' * filled_length + '-' * (length - filled_length)
        self.__verbose(1, " |%s| %s%%" % (bar, percent), end="\r")
        if iteration == total: 
            self.__verbose(1, "\n")

    def fit(self):
        print("Training...")
        for i in range(self.config['epochs']):
            guess = self.__hypothesis(self.x)
            self.__record_cost_evolution(self.__cost(guess))
            gradient = np.dot(self.x.T, guess - self.one_hot_encoded) / self.m
            self.thetas -= self.config['learning_rate'] * gradient
            self.__print_progress_bar(i + 1, self.config['epochs'])
        self.__display_cost_evolution()
        self.__evaluate()
        self.__save()

    def predict(self, save=True):
        prediction = self.__hypothesis(self.x)
        low_level_classes = np.argmax(prediction, axis=1)
        self.df_model_classes = pd.DataFrame(columns=[COLUMN_WITH_CLASSES])
        for i in low_level_classes:
            high_level_class = self.classes[i]
            self.df_model_classes = self.df_model_classes.append({ COLUMN_WITH_CLASSES : high_level_class }, ignore_index=True)
        self.df_model_classes.index.name = 'Index'
        if save:
            self.df_model_classes.to_csv(PREDICTION_FILENAME)
