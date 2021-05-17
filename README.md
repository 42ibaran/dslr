# Data Science and Logistic Regression
AKA DSLR - 42 Project

## Setup
### Locally
You can run the project locally. Make sure you have python installed. I used v3.9, other versions might work too but who knows. To install dependencies run: 
```
pip3 install -r requirements.txt
```

### With Docker
You can also use the project with Docker. First, build the image:
```
docker build -t dslr .
```
Then, you can run the container like so:
```
docker run -it --rm dslr
```
In the container, all dependencies are installed. However, you might have problems with plotting the data due to display forwarding. To make it work (for Mac, not Windows, for Ubuntu see below) make sure that XQuartz is running and connection from remote clients is allowed:

<p align="center">
  <img src="https://raw.githubusercontent.com/42ibaran/ft_linear_regression/master/readme_img/xquartz_setting.png">
</p>

Then on the host run:
```
xhost + 127.0.0.1
```
to allow window forwarding from localhost. That should do it.

On Ubuntu what might work is if you run:
```
xhost + local:root
docker run -it --rm --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" dslr 
```
Not sure about other Linux systems. Sorry 😢

### With VSCode
I kept `.devcontainer` directory with a setup for development using VSCode Remote Development extension. You can reopen the project directory using the extension, similarly to using Docker container but with more functionality.

## Usage
## Data analysis and visualization
### Describe
To display statistics about features in a dataset run:
```
python describe.py [-h] [--to-file] [--output-filename OUTPUT_FILENAME] filename.csv
```

### Histogram
To display the histogram of the score distribution of the most homogeneous feature run:
```
python histogram.py [-h] [--all] filename.csv
```

### Scatter plot
To display the scatter plot of the two most similar features run:
```
python scatter_plot.py [-h] filename.csv
```

### Pair plot
To display the pair plot between all features run:
```
python pair_plot.py [-h] filename.csv
```

## Logistic regression
To train the model run:
```
logreg_train.py [-h] [--cost-evolution] [--epochs EPOCHS] [--learning-rate LEARNING_RATE] [--features FEATURES [FEATURES ...]] [--features-select] [--test] [--split-percent SPLIT_PERCENT] [--seed SEED] [--random] filename.csv
```
Optional arguments:
* ```-c (--cost-evolution)``` display cost evolution after training
* ```-e (--epochs) EPOCHS``` number of iterations for training
* ```-lr (--learning-rate) LEARNING_RATE``` learning rate for training
* ```-f (--features) FEATURE [FEATURES ...]``` list of features for training
* ```-fs (--features-select)``` display feature selector. -f option will be ignored
* ```-t (--test)``` display accuracy statistics after training. will split provided dataset by Pareto rule
* ```-sp (--split-percent) SPLIT_PERCENT``` fraction of dataset to be used for training (see --test)
* ```-s (--seed) SEED``` seed to use for splitting dataset (see --test)
* ```-v (--verbose) VERBOSE``` verbose level

After training is complete, `logreg_train.json` is created with data necessary for prediction.

To sort students into Hogwarts houses (ain't it what we all are here for?) run:
```
python logreg_predict.py datasets/dataset_test.csv
```
File `houses.csv` will be created et voilà!

## Bonuses
1. Verbose
2. Integrated model evaluation
3. Choose epochs, learning rate
4. Choose seed and dataset split ratio
5. Display cost evolution
6. Save describe.py result to file
7. Docker
8. Readme
