import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

keras.utils.set_random_seed(42)
df = pd.read_csv("http://storage.googleapis.com/download.tensorflow.org/data/heart.csv")
df.to_csv("dataset.csv", index=False)




