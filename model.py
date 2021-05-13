import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

from logger import *

DEFAULT_EPOCHS = 2500
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_SPLIT_PERCENT = 0.8
DEFAULT_SEED = 42424242

COLUMN_WITH_CLASSES = 'Hogwarts House'
PICKLE_FILENAME = 'logreg_train.json'
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
        'seed': DEFAULT_SEED,
        'random': False
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

    def __init_for_train(self):
        self.df_to_train = self.df_to_train.dropna().reset_index(drop=True)
        self.x = self.df_to_train.select_dtypes(include=[int, float])

        self.classes = {}
        self.y = self.df_to_train[COLUMN_WITH_CLASSES].astype('category')
        classes_high_level = self.y.unique()
        classes_low_level = self.y.cat.codes.unique()
        self.classes = { int(classes_low_level[i]): classes_high_level[i] for i in range(len(classes_low_level)) }
        self.y = self.y.cat.codes
    
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
        for column in self.normalization:
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
                log.warning("Ignoreing invalid value. Press enter to proceed to training.")        
        if len(indices) == 0:
            self.config['features'] = list(numeric_features)
            return
        indices = [ i - 1 for i in indices ]
        try:
            self.config['features'] = list(numeric_features[indices])
        except IndexError:
            raise ValueError("Somehow you f***ed it up.")

    def __select_features(self):
        if self.config['features'] is not None:
            self.config['features'].append(COLUMN_WITH_CLASSES)
            try:
                self.df_to_train = self.df_to_train[self.config['features']] if self.df_to_train is not None else self.df_to_train
                self.df_to_predict = self.df_to_predict[self.config['features']] if self.df_to_predict is not None else self.df_to_predict
            except KeyError:
                raise Exception("Failed selecting features from dataset.")
            self.config['features'].remove(COLUMN_WITH_CLASSES)

    def __split_dataset(self):
        if self.config['split_percent'] < 0.0 or self.config['split_percent'] > 1.0:
            raise Exception("Invalid dataset split percentage value.")

        if self.config['random']:
            self.config['seed'] = np.random.seed()

        if self.config['test']:
            df_to_train = self.df_to_train.sample(frac=self.config['split_percent'], random_state=self.config['seed'])
            self.df_to_predict = self.df_to_train.drop(df_to_train.index)
            self.df_to_train = df_to_train
            
            self.df_to_predict = self.df_to_predict.reset_index(drop=True)
            self.df_to_train = self.df_to_train.reset_index(drop=True)

    def __configure(self, config):
        for key in self.config.keys():
            if key in config and config[key] is not None:
                self.config[key] = config[key]

    def __record_cost_evolution(self, cost):
        if self.config['cost_evolution']:
            self.cost_evolution.append(cost)

    def __display_cost_evolution(self):
        if not self.config['cost_evolution']:
            return
        ax = sns.lineplot(x=range(self.config['epochs']), y=self.cost_evolution)
        ax.set(xlabel='Epochs', ylabel='Cost')
        plt.show()

    def __load_trained_data(self):
        with open(PICKLE_FILENAME, "r") as fi:
            try:
                obj = json.load(fi)
                self.thetas = np.array(obj['thetas'])
                self.normalization = pd.DataFrame(obj['normalization'])
                self.classes = { int(k): v for k, v in obj['classes'].items() }
                self.config['features'] = obj['features']
            except:
                log.error("Binary file (%s) is invalid." % PICKLE_FILENAME)
                exit(1)

    def __save(self):
        obj = {
            'thetas': self.thetas.tolist(),
            'normalization': self.normalization.to_dict(),
            'classes': self.classes,
            'features': self.config['features']
        }
        with open(PICKLE_FILENAME, "w") as fi:
            json.dump(obj, fi, indent=4)
            log.info("Training result saved successfully.")

    def __test(self):
        if not self.config['test']:
            return

        self.__init_for_predict()
        self.predict()

        if self.df_correct_classes.shape[0] != self.df_model_classes.shape[0]:
            raise Exception("Datasets are not same length.")

        columns = ["Positive/Negative"]
        columns += self.classes.values()
        df_accuracy = pd.DataFrame(columns=columns)
        df_accuracy["Positive/Negative"] = ["True Positive", "True Negative", "False Positive", "False Negative"]
        df_accuracy = df_accuracy.fillna(0)
        df_accuracy.set_index("Positive/Negative", inplace=True)

        self.df_model_classes = self.df_model_classes[COLUMN_WITH_CLASSES]

        for index, correct_class in enumerate(self.df_correct_classes):
            if self.df_model_classes[index] == correct_class:
                df_accuracy[correct_class]["True Positive"] += 1
                df_accuracy.loc["True Negative", df_accuracy.columns != correct_class] += 1
            else:
                df_accuracy[self.df_model_classes[index]]["False Positive"] += 1
                df_accuracy[correct_class]["False Negative"] += 1
        log.info("Accuracy table:")
        print(df_accuracy)
        trues = df_accuracy.loc[["True Positive", "True Negative"]].sum().sum()
        all = df_accuracy.sum().sum()
        accuracy = 100 * trues / all
        log.info("Accuracy: %.2f%%" % accuracy)

    def __hypothesis(self, x):
        return 1 / (1 + np.exp(-np.dot(x, self.thetas)))

    def __cost(self, guess):
        ones = np.ones(self.one_hot_encoded.shape)
        return (-1 / self.m) * np.sum((self.one_hot_encoded * np.log(guess) + (ones - self.one_hot_encoded) * np.log(ones - guess)))

    def fit(self):
        for _ in range(self.config['epochs']):
            guess = self.__hypothesis(self.x)
            self.__record_cost_evolution(self.__cost(guess))
            gradient = np.dot(self.x.T, guess - self.one_hot_encoded) / self.m
            self.thetas -= self.config['learning_rate'] * gradient
        self.__display_cost_evolution()
        self.__test()
        self.__save()

    def predict(self):
        prediction = self.__hypothesis(self.x)
        low_level_classes = np.argmax(prediction, axis=1)
        self.df_model_classes = pd.DataFrame(columns=[COLUMN_WITH_CLASSES])
        for i in low_level_classes:
            high_level_class = self.classes[i]
            self.df_model_classes = self.df_model_classes.append({ COLUMN_WITH_CLASSES : high_level_class }, ignore_index=True)
        self.df_model_classes.index.name = 'Index'
        if self.config['mode'] == MODE_PREDICT:
            self.df_model_classes.to_csv(PREDICTION_FILENAME)
        elif self.config['mode'] != MODE_TRAIN:
            raise Exception("Invalid model initialization mode.")
