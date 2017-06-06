# Random Name Generator Generator
## Introduction
This is a script that will train a LSTM neural network on a list of given names, which can then be used to generate new names that are similar to the given ones. These new names can then be compiled in to a list and randomly drawn from, as is done with the minimal web page in the /docs folder.

## Requirements
* [Python 3.5](https://www.python.org/downloads/release/python-352/)
* [Keras](https://keras.io/)
* Which in turn requires:
  * [TensorFlow](https://www.tensorflow.org/install/)
  * [H5PY](http://www.h5py.org/)
* Note that, if on windows, Python's package manager may fail to install scipy. In this case, it needs to be downloaded and installed manually. The wheel can be downloaded [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scikit-learn). Grab the one that corrosponds to your python version (eg. cp35 for Python 3.5) and operating system (win32 for 32-bit or win_amd64 for 64-bit).

## Usage
Generator.py contains settings at the top of the file. Simply edit these to your preference and run the script. Note that training may take a long time, so it's recommended that you save your model if you wish to generate more names in the future.
