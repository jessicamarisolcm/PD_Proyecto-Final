import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
import os
from keras._tf_keras.keras import *
from matplotlib import pyplot as plt
import seaborn as sns
import math
from tensorflow import keras
import tensorflow_addons as tfa
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageEnhance
import random
import cv2
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')


def get_path(plant_dir:str, dir_test:str):
    if dir_test == 'Test':
        return 'PlantVillage/' + plant_dir + '/Test'
    elif dir_test == 'Train':
        return 'PlantVillage/' + plant_dir + '/Train'
    elif dir_test == 'Valid':
        return 'PlantVillage/' + plant_dir + '/Val'
    
plant_dirs = os.listdir("PlantVillage")
plant_dirs