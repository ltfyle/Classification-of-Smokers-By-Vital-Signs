::: {#109b904ec5eaa8b7 .cell .markdown}
## Little side note from the author
To see a more complete version of the project, go to the project notebook. nbconvert was not working on my computer, so I used an online software. 
Otherwise, enjoy :) Please message me if you any feedback or questions! 
# Classification of Smokers and Nonsmokers by Vital Signs

## Problem Statement

The aim of this project is to classify smokers and non-smokers by vital
signs obtained from a 55k entry dataset collected by the Korean National
Insurance Company.

### Why

Although the negative physical effects of smoking are well known, there
are many reasons why a classification tool for identifying smokers.
Though more research is needed, a tool like the one proposed in this
project could help doctors flag potential smokers earlier, without the
need for self identification which could improve patient outcomes. A
tool like this could also be used to gather population level data about
smokers, which could be used to target environment based interventions.

## Summary

This project did not have the outcome that I had hoped for. The biggest
contributing factor was that the data skewed heavily male (64% vs 34%),
and there was a lack of sufficient data on female smokers (only 4% of
the female data are smokers). With that said, the dataset turned out to
be more about demographic information than vital signs. The highest
accuracy reached was 77.22% using random forest. This almost reaches the
benchmark success rate of 77.36% also using random forest.

## Reflection

If I were to do this project over again, I would control for age, and I
would divide the group into male and female, to control for the numbers
disparity. I would also explore more models.

### Comments

This was my first project. There were some techniques that I didn\`t
know how to use, so I borrowed my methods for finding highly correlated
features and finding influential featuresfrom my [benchmark
notebook](https://github.com/Sofxley/signal-of-smoking-classification).
:::

::: {#facff0927949c7ef .cell .markdown}
# EDA
:::

::: {#ac78925540ecd64e .cell .markdown}
# 1. Imports {#1-imports}
:::

::: {#fd179219e55fd24b .cell .code execution_count="3" ExecuteTime="{\"end_time\":\"2025-04-27T23:44:14.794791Z\",\"start_time\":\"2025-04-27T23:44:14.790319Z\"}"}
``` python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import BaggingClassifier

# Import metrics etc.
from sklearn.model_selection import GridSearchCV, learning_curve, StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold, cross_val_score,cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.metrics import f1_score, recall_score, precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import classification_report
from collections import OrderedDict
from sklearn.feature_selection import RFECV, RFE
from sklearn.linear_model import Perceptron
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import warnings
```
:::

::: {#aba910910fad1e20 .cell .code ExecuteTime="{\"end_time\":\"2025-04-27T23:44:17.582514Z\",\"start_time\":\"2025-04-27T23:44:17.580210Z\"}"}
``` python

SEED = 32

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

# plot style
plt.style.use('Solarize_Light2')
mpl.rc('axes', labelweight='ultralight', titleweight='semibold', labelsize=10, )
```
:::

::: {#c91633730b5fc766 .cell .code ExecuteTime="{\"end_time\":\"2025-04-27T23:44:20.050568Z\",\"start_time\":\"2025-04-27T23:44:20.046971Z\"}"}
``` python
def description(data):
	'''
	Returns a dataframe with a detailed description of the data
	'''
	dtypes = data.dtypes
	counts = data.apply(lambda col: col.count())
	nulls = data.apply(lambda col: col.isnull().sum())
	uniques = data.apply(lambda col: col.unique())
	n_uniques = data.apply(lambda col: col.nunique())
	maxs = data.apply(lambda col: col.max())
	mins = data.apply(lambda col: col.min())

	cols = {'dtypes':dtypes, 'counts':counts, 'nulls' : nulls,
            'max':maxs, 'min':mins,'n_uniques':n_uniques, 'uniques':uniques}
	return pd.DataFrame(data=cols)
```
:::

::: {#9daf55d18af52f18 .cell .markdown}
# 2. Loading the Data {#2-loading-the-data}
:::

::: {#3e9776d254d4d44c .cell .code execution_count="6" ExecuteTime="{\"end_time\":\"2025-04-27T23:44:21.941791Z\",\"start_time\":\"2025-04-27T23:44:21.884635Z\"}"}
``` python
smoking_ = pd.read_csv("smoking.csv", index_col='ID')

smoking = smoking_.copy()
```
:::

:::: {#d071a209a29edc24 .cell .code execution_count="7" ExecuteTime="{\"end_time\":\"2025-04-27T23:44:23.550644Z\",\"start_time\":\"2025-04-27T23:44:23.520456Z\"}"}
``` python
smoking.T
```

::: {.output .execute_result execution_count="7"}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>ID</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>55655</th>
      <th>55663</th>
      <th>55666</th>
      <th>55671</th>
      <th>55673</th>
      <th>55676</th>
      <th>55681</th>
      <th>55683</th>
      <th>55684</th>
      <th>55691</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gender</th>
      <td>F</td>
      <td>F</td>
      <td>M</td>
      <td>M</td>
      <td>F</td>
      <td>M</td>
      <td>M</td>
      <td>M</td>
      <td>F</td>
      <td>M</td>
      <td>...</td>
      <td>M</td>
      <td>M</td>
      <td>M</td>
      <td>M</td>
      <td>F</td>
      <td>F</td>
      <td>F</td>
      <td>F</td>
      <td>M</td>
      <td>M</td>
    </tr>
    <tr>
      <th>age</th>
      <td>40</td>
      <td>40</td>
      <td>55</td>
      <td>40</td>
      <td>40</td>
      <td>30</td>
      <td>40</td>
      <td>45</td>
      <td>50</td>
      <td>45</td>
      <td>...</td>
      <td>20</td>
      <td>40</td>
      <td>40</td>
      <td>40</td>
      <td>60</td>
      <td>40</td>
      <td>45</td>
      <td>55</td>
      <td>60</td>
      <td>55</td>
    </tr>
    <tr>
      <th>height(cm)</th>
      <td>155</td>
      <td>160</td>
      <td>170</td>
      <td>165</td>
      <td>155</td>
      <td>180</td>
      <td>160</td>
      <td>165</td>
      <td>150</td>
      <td>175</td>
      <td>...</td>
      <td>175</td>
      <td>180</td>
      <td>170</td>
      <td>170</td>
      <td>150</td>
      <td>170</td>
      <td>160</td>
      <td>160</td>
      <td>165</td>
      <td>160</td>
    </tr>
    <tr>
      <th>weight(kg)</th>
      <td>60</td>
      <td>60</td>
      <td>60</td>
      <td>70</td>
      <td>60</td>
      <td>75</td>
      <td>60</td>
      <td>90</td>
      <td>60</td>
      <td>75</td>
      <td>...</td>
      <td>75</td>
      <td>85</td>
      <td>65</td>
      <td>80</td>
      <td>50</td>
      <td>65</td>
      <td>50</td>
      <td>50</td>
      <td>60</td>
      <td>65</td>
    </tr>
    <tr>
      <th>waist(cm)</th>
      <td>81.3</td>
      <td>81.0</td>
      <td>80.0</td>
      <td>88.0</td>
      <td>86.0</td>
      <td>85.0</td>
      <td>85.5</td>
      <td>96.0</td>
      <td>85.0</td>
      <td>89.0</td>
      <td>...</td>
      <td>85.0</td>
      <td>86.5</td>
      <td>85.0</td>
      <td>90.5</td>
      <td>75.0</td>
      <td>75.0</td>
      <td>70.0</td>
      <td>68.5</td>
      <td>78.0</td>
      <td>85.0</td>
    </tr>
    <tr>
      <th>eyesight(left)</th>
      <td>1.2</td>
      <td>0.8</td>
      <td>0.8</td>
      <td>1.5</td>
      <td>1.0</td>
      <td>1.2</td>
      <td>1.0</td>
      <td>1.2</td>
      <td>0.7</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.9</td>
      <td>1.2</td>
      <td>1.2</td>
      <td>1.2</td>
      <td>1.0</td>
      <td>0.9</td>
      <td>1.2</td>
      <td>1.0</td>
      <td>0.8</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>eyesight(right)</th>
      <td>1.0</td>
      <td>0.6</td>
      <td>0.8</td>
      <td>1.5</td>
      <td>1.0</td>
      <td>1.2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.8</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.5</td>
      <td>1.2</td>
      <td>1.2</td>
      <td>1.5</td>
      <td>1.2</td>
      <td>0.9</td>
      <td>1.2</td>
      <td>1.2</td>
      <td>1.0</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>hearing(left)</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>hearing(right)</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>systolic</th>
      <td>114.0</td>
      <td>119.0</td>
      <td>138.0</td>
      <td>100.0</td>
      <td>120.0</td>
      <td>128.0</td>
      <td>116.0</td>
      <td>153.0</td>
      <td>115.0</td>
      <td>113.0</td>
      <td>...</td>
      <td>118.0</td>
      <td>116.0</td>
      <td>106.0</td>
      <td>130.0</td>
      <td>102.0</td>
      <td>110.0</td>
      <td>101.0</td>
      <td>117.0</td>
      <td>133.0</td>
      <td>124.0</td>
    </tr>
    <tr>
      <th>relaxation</th>
      <td>73.0</td>
      <td>70.0</td>
      <td>86.0</td>
      <td>60.0</td>
      <td>74.0</td>
      <td>76.0</td>
      <td>82.0</td>
      <td>96.0</td>
      <td>74.0</td>
      <td>64.0</td>
      <td>...</td>
      <td>72.0</td>
      <td>69.0</td>
      <td>69.0</td>
      <td>84.0</td>
      <td>60.0</td>
      <td>68.0</td>
      <td>62.0</td>
      <td>72.0</td>
      <td>76.0</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>fasting blood sugar</th>
      <td>94.0</td>
      <td>130.0</td>
      <td>89.0</td>
      <td>96.0</td>
      <td>80.0</td>
      <td>95.0</td>
      <td>94.0</td>
      <td>158.0</td>
      <td>86.0</td>
      <td>94.0</td>
      <td>...</td>
      <td>80.0</td>
      <td>96.0</td>
      <td>85.0</td>
      <td>91.0</td>
      <td>85.0</td>
      <td>89.0</td>
      <td>89.0</td>
      <td>88.0</td>
      <td>107.0</td>
      <td>82.0</td>
    </tr>
    <tr>
      <th>Cholesterol</th>
      <td>215.0</td>
      <td>192.0</td>
      <td>242.0</td>
      <td>322.0</td>
      <td>184.0</td>
      <td>217.0</td>
      <td>226.0</td>
      <td>222.0</td>
      <td>210.0</td>
      <td>198.0</td>
      <td>...</td>
      <td>167.0</td>
      <td>289.0</td>
      <td>192.0</td>
      <td>216.0</td>
      <td>179.0</td>
      <td>213.0</td>
      <td>166.0</td>
      <td>158.0</td>
      <td>210.0</td>
      <td>213.0</td>
    </tr>
    <tr>
      <th>triglyceride</th>
      <td>82.0</td>
      <td>115.0</td>
      <td>182.0</td>
      <td>254.0</td>
      <td>74.0</td>
      <td>199.0</td>
      <td>68.0</td>
      <td>269.0</td>
      <td>66.0</td>
      <td>147.0</td>
      <td>...</td>
      <td>167.0</td>
      <td>150.0</td>
      <td>162.0</td>
      <td>121.0</td>
      <td>53.0</td>
      <td>99.0</td>
      <td>69.0</td>
      <td>77.0</td>
      <td>79.0</td>
      <td>142.0</td>
    </tr>
    <tr>
      <th>HDL</th>
      <td>73.0</td>
      <td>42.0</td>
      <td>55.0</td>
      <td>45.0</td>
      <td>62.0</td>
      <td>48.0</td>
      <td>55.0</td>
      <td>34.0</td>
      <td>48.0</td>
      <td>43.0</td>
      <td>...</td>
      <td>53.0</td>
      <td>68.0</td>
      <td>44.0</td>
      <td>57.0</td>
      <td>52.0</td>
      <td>75.0</td>
      <td>73.0</td>
      <td>79.0</td>
      <td>48.0</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>LDL</th>
      <td>126.0</td>
      <td>127.0</td>
      <td>151.0</td>
      <td>226.0</td>
      <td>107.0</td>
      <td>129.0</td>
      <td>157.0</td>
      <td>134.0</td>
      <td>149.0</td>
      <td>126.0</td>
      <td>...</td>
      <td>80.0</td>
      <td>183.0</td>
      <td>116.0</td>
      <td>135.0</td>
      <td>116.0</td>
      <td>118.0</td>
      <td>79.0</td>
      <td>63.0</td>
      <td>146.0</td>
      <td>150.0</td>
    </tr>
    <tr>
      <th>hemoglobin</th>
      <td>12.9</td>
      <td>12.7</td>
      <td>15.8</td>
      <td>14.7</td>
      <td>12.5</td>
      <td>16.2</td>
      <td>17.0</td>
      <td>15.0</td>
      <td>13.7</td>
      <td>16.0</td>
      <td>...</td>
      <td>16.6</td>
      <td>16.3</td>
      <td>15.6</td>
      <td>14.8</td>
      <td>12.6</td>
      <td>12.3</td>
      <td>14.0</td>
      <td>12.4</td>
      <td>14.4</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>Urine protein</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>serum creatinine</th>
      <td>0.7</td>
      <td>0.6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.6</td>
      <td>1.2</td>
      <td>0.7</td>
      <td>1.3</td>
      <td>0.8</td>
      <td>0.8</td>
      <td>...</td>
      <td>1.2</td>
      <td>1.3</td>
      <td>1.1</td>
      <td>0.9</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>0.9</td>
      <td>0.5</td>
      <td>0.7</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>AST</th>
      <td>18.0</td>
      <td>22.0</td>
      <td>21.0</td>
      <td>19.0</td>
      <td>16.0</td>
      <td>18.0</td>
      <td>21.0</td>
      <td>38.0</td>
      <td>31.0</td>
      <td>26.0</td>
      <td>...</td>
      <td>13.0</td>
      <td>21.0</td>
      <td>22.0</td>
      <td>16.0</td>
      <td>26.0</td>
      <td>14.0</td>
      <td>20.0</td>
      <td>17.0</td>
      <td>20.0</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>ALT</th>
      <td>19.0</td>
      <td>19.0</td>
      <td>16.0</td>
      <td>26.0</td>
      <td>14.0</td>
      <td>27.0</td>
      <td>27.0</td>
      <td>71.0</td>
      <td>31.0</td>
      <td>24.0</td>
      <td>...</td>
      <td>6.0</td>
      <td>19.0</td>
      <td>25.0</td>
      <td>28.0</td>
      <td>21.0</td>
      <td>7.0</td>
      <td>12.0</td>
      <td>11.0</td>
      <td>19.0</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>Gtp</th>
      <td>27.0</td>
      <td>18.0</td>
      <td>22.0</td>
      <td>18.0</td>
      <td>22.0</td>
      <td>33.0</td>
      <td>39.0</td>
      <td>111.0</td>
      <td>14.0</td>
      <td>63.0</td>
      <td>...</td>
      <td>14.0</td>
      <td>38.0</td>
      <td>33.0</td>
      <td>68.0</td>
      <td>14.0</td>
      <td>10.0</td>
      <td>14.0</td>
      <td>12.0</td>
      <td>18.0</td>
      <td>41.0</td>
    </tr>
    <tr>
      <th>oral</th>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>...</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>dental caries</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>tartar</th>
      <td>Y</td>
      <td>Y</td>
      <td>N</td>
      <td>Y</td>
      <td>N</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>N</td>
      <td>N</td>
      <td>...</td>
      <td>Y</td>
      <td>N</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>N</td>
      <td>N</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>smoking</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>26 rows × 55692 columns</p>
</div>
:::
::::

::: {#780fa4f527247e81 .cell .code execution_count="8"}
``` python
smoking = smoking.drop("oral", axis = 1)
```
:::

::: {#3302651317524bff .cell .markdown}
I dropped oral because all the answers were \"Y\", so it did not add
anything to the data.
:::

:::: {#54ddc13727f8f18c .cell .code execution_count="9" ExecuteTime="{\"end_time\":\"2025-04-27T23:44:28.597687Z\",\"start_time\":\"2025-04-27T23:44:28.594163Z\"}"}
``` python
# Change the column names to the more convinient ones. copied this from the benchmark notebook.
smoking.rename(columns={'height(cm)':'height', 'weight(kg)':'weight','waist(cm)':'waist',
                        'eyesight(left)':'eyesight_left', 'eyesight(right)':'eyesight_right',
                        'hearing(left)':'hearing_left', 'hearing(right)':'hearing_right',
                        'fasting blood sugar':'fasting_blood_sugar',  'Cholesterol':'cholesterol',
                        'HDL':'hdl','LDL':'ldl','Urine protein':'urine_protein',
                        'serum creatinine':'serum_creatinine', 'AST':'ast','ALT':'alt',
                        'Gtp':'gtp', 'dental caries' : 'dental_caries'}, inplace=True)

smoking.shape
```

::: {.output .execute_result execution_count="9"}
    (55692, 25)
:::
::::

:::: {#8b2628c05374f75f .cell .code execution_count="10" ExecuteTime="{\"end_time\":\"2025-04-27T23:44:30.685596Z\",\"start_time\":\"2025-04-27T23:44:30.676219Z\"}"}
``` python
smoking.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    Index: 55692 entries, 0 to 55691
    Data columns (total 25 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   gender               55692 non-null  object 
     1   age                  55692 non-null  int64  
     2   height               55692 non-null  int64  
     3   weight               55692 non-null  int64  
     4   waist                55692 non-null  float64
     5   eyesight_left        55692 non-null  float64
     6   eyesight_right       55692 non-null  float64
     7   hearing_left         55692 non-null  float64
     8   hearing_right        55692 non-null  float64
     9   systolic             55692 non-null  float64
     10  relaxation           55692 non-null  float64
     11  fasting_blood_sugar  55692 non-null  float64
     12  cholesterol          55692 non-null  float64
     13  triglyceride         55692 non-null  float64
     14  hdl                  55692 non-null  float64
     15  ldl                  55692 non-null  float64
     16  hemoglobin           55692 non-null  float64
     17  urine_protein        55692 non-null  float64
     18  serum_creatinine     55692 non-null  float64
     19  ast                  55692 non-null  float64
     20  alt                  55692 non-null  float64
     21  gtp                  55692 non-null  float64
     22  dental_caries        55692 non-null  int64  
     23  tartar               55692 non-null  object 
     24  smoking              55692 non-null  int64  
    dtypes: float64(18), int64(5), object(2)
    memory usage: 13.1+ MB
:::
::::

::: {#264d509c2c9e0d3a .cell .markdown}
Useful function. Reveales that there are no missing values. It also
yields clues as to continuous vs discreet data, and categorical data.
:::

:::: {#a20da1f916f84e02 .cell .code execution_count="11" ExecuteTime="{\"end_time\":\"2025-04-27T23:44:33.210362Z\",\"start_time\":\"2025-04-27T23:44:33.166310Z\"}"}
``` python
description(smoking)
```

::: {.output .execute_result execution_count="11"}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dtypes</th>
      <th>counts</th>
      <th>nulls</th>
      <th>max</th>
      <th>min</th>
      <th>n_uniques</th>
      <th>uniques</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gender</th>
      <td>object</td>
      <td>55692</td>
      <td>0</td>
      <td>M</td>
      <td>F</td>
      <td>2</td>
      <td>[F, M]</td>
    </tr>
    <tr>
      <th>age</th>
      <td>int64</td>
      <td>55692</td>
      <td>0</td>
      <td>85</td>
      <td>20</td>
      <td>14</td>
      <td>[40, 55, 30, 45, 50, 35, 60, 25, 65, 20, 80, 7...</td>
    </tr>
    <tr>
      <th>height</th>
      <td>int64</td>
      <td>55692</td>
      <td>0</td>
      <td>190</td>
      <td>130</td>
      <td>13</td>
      <td>[155, 160, 170, 165, 180, 150, 175, 140, 185, ...</td>
    </tr>
    <tr>
      <th>weight</th>
      <td>int64</td>
      <td>55692</td>
      <td>0</td>
      <td>135</td>
      <td>30</td>
      <td>22</td>
      <td>[60, 70, 75, 90, 65, 45, 55, 50, 85, 80, 100, ...</td>
    </tr>
    <tr>
      <th>waist</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>129.0</td>
      <td>51.0</td>
      <td>566</td>
      <td>[81.3, 81.0, 80.0, 88.0, 86.0, 85.0, 85.5, 96....</td>
    </tr>
    <tr>
      <th>eyesight_left</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>9.9</td>
      <td>0.1</td>
      <td>19</td>
      <td>[1.2, 0.8, 1.5, 1.0, 0.7, 0.9, 0.3, 0.2, 0.1, ...</td>
    </tr>
    <tr>
      <th>eyesight_right</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>9.9</td>
      <td>0.1</td>
      <td>17</td>
      <td>[1.0, 0.6, 0.8, 1.5, 1.2, 0.7, 0.4, 0.9, 0.3, ...</td>
    </tr>
    <tr>
      <th>hearing_left</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>[1.0, 2.0]</td>
    </tr>
    <tr>
      <th>hearing_right</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>[1.0, 2.0]</td>
    </tr>
    <tr>
      <th>systolic</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>240.0</td>
      <td>71.0</td>
      <td>130</td>
      <td>[114.0, 119.0, 138.0, 100.0, 120.0, 128.0, 116...</td>
    </tr>
    <tr>
      <th>relaxation</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>146.0</td>
      <td>40.0</td>
      <td>95</td>
      <td>[73.0, 70.0, 86.0, 60.0, 74.0, 76.0, 82.0, 96....</td>
    </tr>
    <tr>
      <th>fasting_blood_sugar</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>505.0</td>
      <td>46.0</td>
      <td>276</td>
      <td>[94.0, 130.0, 89.0, 96.0, 80.0, 95.0, 158.0, 8...</td>
    </tr>
    <tr>
      <th>cholesterol</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>445.0</td>
      <td>55.0</td>
      <td>286</td>
      <td>[215.0, 192.0, 242.0, 322.0, 184.0, 217.0, 226...</td>
    </tr>
    <tr>
      <th>triglyceride</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>999.0</td>
      <td>8.0</td>
      <td>390</td>
      <td>[82.0, 115.0, 182.0, 254.0, 74.0, 199.0, 68.0,...</td>
    </tr>
    <tr>
      <th>hdl</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>618.0</td>
      <td>4.0</td>
      <td>126</td>
      <td>[73.0, 42.0, 55.0, 45.0, 62.0, 48.0, 34.0, 43....</td>
    </tr>
    <tr>
      <th>ldl</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>1860.0</td>
      <td>1.0</td>
      <td>289</td>
      <td>[126.0, 127.0, 151.0, 226.0, 107.0, 129.0, 157...</td>
    </tr>
    <tr>
      <th>hemoglobin</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>21.1</td>
      <td>4.9</td>
      <td>145</td>
      <td>[12.9, 12.7, 15.8, 14.7, 12.5, 16.2, 17.0, 15....</td>
    </tr>
    <tr>
      <th>urine_protein</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>6</td>
      <td>[1.0, 3.0, 2.0, 4.0, 5.0, 6.0]</td>
    </tr>
    <tr>
      <th>serum_creatinine</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>11.6</td>
      <td>0.1</td>
      <td>38</td>
      <td>[0.7, 0.6, 1.0, 1.2, 1.3, 0.8, 1.1, 0.9, 0.5, ...</td>
    </tr>
    <tr>
      <th>ast</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>1311.0</td>
      <td>6.0</td>
      <td>219</td>
      <td>[18.0, 22.0, 21.0, 19.0, 16.0, 38.0, 31.0, 26....</td>
    </tr>
    <tr>
      <th>alt</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>2914.0</td>
      <td>1.0</td>
      <td>245</td>
      <td>[19.0, 16.0, 26.0, 14.0, 27.0, 71.0, 31.0, 24....</td>
    </tr>
    <tr>
      <th>gtp</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>999.0</td>
      <td>1.0</td>
      <td>488</td>
      <td>[27.0, 18.0, 22.0, 33.0, 39.0, 111.0, 14.0, 63...</td>
    </tr>
    <tr>
      <th>dental_caries</th>
      <td>int64</td>
      <td>55692</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>[0, 1]</td>
    </tr>
    <tr>
      <th>tartar</th>
      <td>object</td>
      <td>55692</td>
      <td>0</td>
      <td>Y</td>
      <td>N</td>
      <td>2</td>
      <td>[Y, N]</td>
    </tr>
    <tr>
      <th>smoking</th>
      <td>int64</td>
      <td>55692</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>[0, 1]</td>
    </tr>
  </tbody>
</table>
</div>
:::
::::

::: {#3502be80cdc8e7a3 .cell .markdown}
This reveals in addition to the target variable, there are 3 categorical
variables: gender, dental_caries and tartar. There are also 3 ordinal
categorical variables: hearing_left, hearing_right and urine_protein.
:::

::: {#92f6af70f3e86504 .cell .markdown}
# Encoding Gender And Tartar
:::

:::: {#63ab264d096c71c0 .cell .code execution_count="12" ExecuteTime="{\"end_time\":\"2025-04-27T23:44:37.597987Z\",\"start_time\":\"2025-04-27T23:44:37.557955Z\"}"}
``` python
for col in ['gender','tartar']:
	smoking[col] = LabelEncoder().fit_transform(smoking[col])

description(smoking)
```

::: {.output .execute_result execution_count="12"}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dtypes</th>
      <th>counts</th>
      <th>nulls</th>
      <th>max</th>
      <th>min</th>
      <th>n_uniques</th>
      <th>uniques</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gender</th>
      <td>int64</td>
      <td>55692</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>[0, 1]</td>
    </tr>
    <tr>
      <th>age</th>
      <td>int64</td>
      <td>55692</td>
      <td>0</td>
      <td>85.0</td>
      <td>20.0</td>
      <td>14</td>
      <td>[40, 55, 30, 45, 50, 35, 60, 25, 65, 20, 80, 7...</td>
    </tr>
    <tr>
      <th>height</th>
      <td>int64</td>
      <td>55692</td>
      <td>0</td>
      <td>190.0</td>
      <td>130.0</td>
      <td>13</td>
      <td>[155, 160, 170, 165, 180, 150, 175, 140, 185, ...</td>
    </tr>
    <tr>
      <th>weight</th>
      <td>int64</td>
      <td>55692</td>
      <td>0</td>
      <td>135.0</td>
      <td>30.0</td>
      <td>22</td>
      <td>[60, 70, 75, 90, 65, 45, 55, 50, 85, 80, 100, ...</td>
    </tr>
    <tr>
      <th>waist</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>129.0</td>
      <td>51.0</td>
      <td>566</td>
      <td>[81.3, 81.0, 80.0, 88.0, 86.0, 85.0, 85.5, 96....</td>
    </tr>
    <tr>
      <th>eyesight_left</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>9.9</td>
      <td>0.1</td>
      <td>19</td>
      <td>[1.2, 0.8, 1.5, 1.0, 0.7, 0.9, 0.3, 0.2, 0.1, ...</td>
    </tr>
    <tr>
      <th>eyesight_right</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>9.9</td>
      <td>0.1</td>
      <td>17</td>
      <td>[1.0, 0.6, 0.8, 1.5, 1.2, 0.7, 0.4, 0.9, 0.3, ...</td>
    </tr>
    <tr>
      <th>hearing_left</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>[1.0, 2.0]</td>
    </tr>
    <tr>
      <th>hearing_right</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>[1.0, 2.0]</td>
    </tr>
    <tr>
      <th>systolic</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>240.0</td>
      <td>71.0</td>
      <td>130</td>
      <td>[114.0, 119.0, 138.0, 100.0, 120.0, 128.0, 116...</td>
    </tr>
    <tr>
      <th>relaxation</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>146.0</td>
      <td>40.0</td>
      <td>95</td>
      <td>[73.0, 70.0, 86.0, 60.0, 74.0, 76.0, 82.0, 96....</td>
    </tr>
    <tr>
      <th>fasting_blood_sugar</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>505.0</td>
      <td>46.0</td>
      <td>276</td>
      <td>[94.0, 130.0, 89.0, 96.0, 80.0, 95.0, 158.0, 8...</td>
    </tr>
    <tr>
      <th>cholesterol</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>445.0</td>
      <td>55.0</td>
      <td>286</td>
      <td>[215.0, 192.0, 242.0, 322.0, 184.0, 217.0, 226...</td>
    </tr>
    <tr>
      <th>triglyceride</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>999.0</td>
      <td>8.0</td>
      <td>390</td>
      <td>[82.0, 115.0, 182.0, 254.0, 74.0, 199.0, 68.0,...</td>
    </tr>
    <tr>
      <th>hdl</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>618.0</td>
      <td>4.0</td>
      <td>126</td>
      <td>[73.0, 42.0, 55.0, 45.0, 62.0, 48.0, 34.0, 43....</td>
    </tr>
    <tr>
      <th>ldl</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>1860.0</td>
      <td>1.0</td>
      <td>289</td>
      <td>[126.0, 127.0, 151.0, 226.0, 107.0, 129.0, 157...</td>
    </tr>
    <tr>
      <th>hemoglobin</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>21.1</td>
      <td>4.9</td>
      <td>145</td>
      <td>[12.9, 12.7, 15.8, 14.7, 12.5, 16.2, 17.0, 15....</td>
    </tr>
    <tr>
      <th>urine_protein</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>6</td>
      <td>[1.0, 3.0, 2.0, 4.0, 5.0, 6.0]</td>
    </tr>
    <tr>
      <th>serum_creatinine</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>11.6</td>
      <td>0.1</td>
      <td>38</td>
      <td>[0.7, 0.6, 1.0, 1.2, 1.3, 0.8, 1.1, 0.9, 0.5, ...</td>
    </tr>
    <tr>
      <th>ast</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>1311.0</td>
      <td>6.0</td>
      <td>219</td>
      <td>[18.0, 22.0, 21.0, 19.0, 16.0, 38.0, 31.0, 26....</td>
    </tr>
    <tr>
      <th>alt</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>2914.0</td>
      <td>1.0</td>
      <td>245</td>
      <td>[19.0, 16.0, 26.0, 14.0, 27.0, 71.0, 31.0, 24....</td>
    </tr>
    <tr>
      <th>gtp</th>
      <td>float64</td>
      <td>55692</td>
      <td>0</td>
      <td>999.0</td>
      <td>1.0</td>
      <td>488</td>
      <td>[27.0, 18.0, 22.0, 33.0, 39.0, 111.0, 14.0, 63...</td>
    </tr>
    <tr>
      <th>dental_caries</th>
      <td>int64</td>
      <td>55692</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>[0, 1]</td>
    </tr>
    <tr>
      <th>tartar</th>
      <td>int64</td>
      <td>55692</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>[1, 0]</td>
    </tr>
    <tr>
      <th>smoking</th>
      <td>int64</td>
      <td>55692</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>[0, 1]</td>
    </tr>
  </tbody>
</table>
</div>
:::
::::

::: {#d76b4316ccbe1103 .cell .markdown}
# 3. EDA {#3-eda}
:::

::::: {#f64c90323a5f2fe .cell .code execution_count="13" ExecuteTime="{\"end_time\":\"2025-04-27T23:44:41.281840Z\",\"start_time\":\"2025-04-27T23:44:40.709352Z\"}"}
``` python
plt.figure(figsize=(15,12))

sns.heatmap(smoking.corr(), annot=True, cmap='cubehelix', fmt='.2f',)
plt.title('Correlation Plot')
```

::: {.output .execute_result execution_count="13"}
    Text(0.5, 1.0, 'Correlation Plot')
:::

::: {.output .display_data}
![](af174c03b35800167ce424e0006f7b9a9458a7f0.png)
:::
:::::

::: {#f4b2f8457b77af8b .cell .markdown}
## Correlation Summary

- **Gender**
  - Strongly correlated with:
    1.  Height
    2.  Hemoglobin
    3.  Weight
    4.  Serum Creatinine
- **Height**
  - Strongly correlated with:
    1.  Gender
    2.  Weight
    3.  Hemoglobin
- **Weight**
  - Strongly correlated with:
    1.  Waist circumference
    2.  Height
    3.  Gender
- **Other Notable Pairs**
  - **Systolic Blood Pressure** & **Diastolic (Relaxation) Blood
    Pressure**
  - **AST** & **ALT** (Liver Enzymes)
  - **LDL** & **Total Cholesterol**
  - **Hearing (Left)** & **Hearing (Right)**
  - **Eyesight (Left)** & **Eyesight (Right)**
:::

::: {#d4d238674d5bfe35 .cell .markdown}
We should note that gender and smoking have a high correlation at 0.51.

- We should expect high correlation between gender & hemoglobin and
  gender & serum creatine. This is backed up by common medical
  knowledge.
- None of the other correlations listed here are unexpected either.
:::

::: {#4749065d1d7d0108 .cell .markdown}
# Segmented Analysis of Smoking And Gender
:::

::: {#4253adc282ae876c .cell .code execution_count="15" ExecuteTime="{\"end_time\":\"2025-04-27T23:44:48.754803Z\",\"start_time\":\"2025-04-27T23:44:48.744744Z\"}"}
``` python
smokers= smoking.loc[smoking['smoking']==1]
non_smokers= smoking.loc[smoking['smoking']==0]


male = smoking.loc[smoking['gender']==1]
female = smoking.loc[smoking['gender']==0]

# there is an option to specify columns

```
:::

::::: {#a1ed66a36b9d2a74 .cell .code execution_count="16" ExecuteTime="{\"end_time\":\"2025-04-27T23:44:50.656011Z\",\"start_time\":\"2025-04-27T23:44:50.505734Z\"}"}
``` python
plt.figure(figsize=(10, 10))

plt.subplot(321)
plt.pie(smoking['smoking'].value_counts(), labels=['non-smoking', 'smoking'],
		autopct="%1.2f%%", colors=["#5F9EA0", "#ADD8E6"],
		wedgeprops=dict(width=1, edgecolor='w', linewidth=2), shadow=True, )
plt.title('Smoking Habit', fontsize=14)

plt.subplot(322)
plt.pie(smoking['gender'].value_counts(), labels=['Male','Female'],
		autopct="%1.2f%%", colors = ["#B7410E", "#FFA500"],
		wedgeprops=dict(width=1, edgecolor='white', linewidth=2), shadow=True, )
plt.title('Gender', fontsize=14, )

plt.subplot(323)

plt.pie(smokers['gender'].value_counts(), labels=['Male','Female'],
        autopct="%1.2f%%", colors = ["#D5006D", "#FFD700"],
        wedgeprops=dict(width=1, edgecolor='white', linewidth=2), shadow=True, )
plt.title('Smokers by Gender', fontsize=14, )


plt.subplot(324)
plt.pie(non_smokers['gender'].value_counts(), labels=['Male','Female'],
        autopct="%1.2f%%", colors = ["#D2386C", "#FF7F11"],
        wedgeprops=dict(width=1, edgecolor='white', linewidth=2), shadow=True, )
plt.title('Non-Smokers by Gender', fontsize=14, )

plt.subplot(325)
plt.pie(male['smoking'].value_counts(), labels=['Non-Smokers','Smokers'],
        autopct="%1.2f%%", colors=["#8093f1", "#e7c6ff"],
        wedgeprops=dict(width=1, edgecolor='white', linewidth=2), shadow=True, )
plt.title('Male Smoker Classification', fontsize=14, )

plt.subplot(326)
plt.pie(female['smoking'].value_counts(), labels=['Non-Smokers','Smokers'],
        autopct="%1.2f%%", colors=["#8093f1", "#e7c6ff"],
        wedgeprops=dict(width=1, edgecolor='white', linewidth=2), shadow=True, )
plt.title('Female Smoker Classification', fontsize=14, )
```

::: {.output .execute_result execution_count="16"}
    Text(0.5, 1.0, 'Female Smoker Classification')
:::

::: {.output .display_data}
![](1a15c8a7ae10f947e5640a00d269f6788e5bb256.png)
:::
:::::

::: {#715aa805929d3b09 .cell .markdown}
Our data is highly i
:::

:::: {#ad884c294b7ce9b4 .cell .code execution_count="17" ExecuteTime="{\"end_time\":\"2025-04-27T23:44:54.619097Z\",\"start_time\":\"2025-04-27T23:44:54.616243Z\"}"}
``` python
print(smoking['gender'].value_counts())
```

::: {.output .stream .stdout}
    gender
    1    35401
    0    20291
    Name: count, dtype: int64
:::
::::

::: {#dd3eb9c28861ebec .cell .markdown}
At first, I was alarmed by the results of my segmentation analysis. Only
4.23% of females are smokers and they only make up 4.20% of the smoking
population. This is a problem because our model could be influenced by
this bias.

The easiest way to deal with this given my current capabilities would be
to just change my problem definition slightly, to focus on the
classification of male smokers by vital signs, since the male population
is fairly balanced between smokers and non-smokers, and we have a large
dataset. I may end up creating three models, one for male, female and
the whole population since I have no idea how to deal with such a large
bias.
:::

::: {#990e6363e305016d .cell .markdown}
## Age Breakdown
:::

::::: {#99d2a6e8ecbc9a65 .cell .code execution_count="18" ExecuteTime="{\"end_time\":\"2025-04-27T23:44:57.984408Z\",\"start_time\":\"2025-04-27T23:44:56.784646Z\"}"}
``` python
# Age w. respect to smoking status
# Age w. respect to gender
fig = plt.figure(figsize=(10,10))

plt.subplot(221)
sns.kdeplot(data=smoking, x='age', hue='smoking', palette='hls', linewidth=3, fill=True,)
plt.title('Distribution of Age-Smoking')
plt.legend(labels=['yes','no'], title='smoking')

plt.subplot(222)
sns.kdeplot(data=smoking, x='age', hue='gender', palette='hls', linewidth=3, fill=True,)
plt.title('Distribution of Age-Gender')
plt.legend(title='Gender', labels=['Male','Female'])

# women smoking, age
plt.subplot(223)
sns.kdeplot(data=female, x='age', hue='smoking', palette='hls', linewidth=3, fill=True,)
plt.legend(title='Smoking', labels=['yes','no'])
plt.title("Distribution of Age,Smoking- Female")

plt.subplot(224)
sns.kdeplot(data=male, x='age', hue='smoking', palette='hls', linewidth=3, fill=True,)
plt.legend(title='Smoking', labels=['yes','no'])
plt.title("Distribution of Age,Smoking- Male")
# men smoking, age
```

::: {.output .execute_result execution_count="18"}
    Text(0.5, 1.0, 'Distribution of Age,Smoking- Male')
:::

::: {.output .display_data}
![](4e9406d533ec785eaa8f9c00f3760b02829836b7.png)
:::
:::::

::: {#b2675fd620620fe5 .cell .markdown}
The age distribution for this study is fairly balanced for men, but
shows a much older demographic for women. This could be a result of how
the data was collected.

The study shows an average age for participants and smokers alike to be
40. The age demographic of the women could be a reason so few women
reported smoking. It could be possible that cultural norms have shifted
over the years towards less taboos towards women engaging in \"vices\".

It is likely that smoking is most common regardless of gender, among the
age demographic of people around 40 years old. Perhaps these people were
the right age when public knowledge around smoking was lax, and smoking
was fashionable.
:::

::: {#b42f31b7e14f8d39 .cell .markdown}
# Exploration of Continuous Variables by Age, Gender and Smoking Status

Here are some violin plots exploring a few continuous variables. I have
them tested for age and gender on the right, and smoking and gender on
the left. I chose to do this since it was a clear middle aged
demographic for smokers, and I wanted to check that the statistical
signifficance could not be explained away by age.
:::

::: {#f4e4af58ec28d7e7 .cell .code execution_count="20" ExecuteTime="{\"end_time\":\"2025-04-27T23:45:02.785359Z\",\"start_time\":\"2025-04-27T23:45:02.782681Z\"}"}
``` python
cols = ['hemoglobin','weight','waist','serum_creatinine','ast','alt','ldl','cholesterol']
```
:::

:::: {#23a56a8f35e2b871 .cell .code execution_count="21" ExecuteTime="{\"end_time\":\"2025-04-27T23:45:08.310600Z\",\"start_time\":\"2025-04-27T23:45:04.372490Z\"}"}
``` python
warnings.filterwarnings('ignore')

fig = plt.figure(figsize=(20,50))

for i, col in enumerate(cols):

    plt.subplot(len(cols), 2, i*2 + 1)
    sns.violinplot(
        data=smoking,
        x='age',
        y=col,
        hue='gender',
        split=False,
        native_scale=True,
        legend='full',
        palette='coolwarm',
        inner=None,
	    linewidth=0.1,
	    #linecolor='dark grey'

    )

    plt.title(f'Age vs {col}')
    plt.xlabel('Age')
    plt.ylabel(col)
    plt.legend(title='Gender', loc='best')

    handles, labels = plt.gca().get_legend_handles_labels()
    custom_labels = ['Female','Male']
    plt.legend(handles, custom_labels, title='Gender', loc='upper right')


    ax1 = plt.gca()
    ax1.set_facecolor("snow")


    plt.subplot(len(cols), 2, i*2 + 2)
    sns.violinplot(
        data=smoking,
        x=col,
        y='gender',
        hue='smoking',
        split=True,
        #inner="quart",
        inner_kws=dict(box_width=10, whis_width=2, color=".5"),
        gap = .1,
        orient='h',
        bw=0.25,
        cut=0,
        legend='full',
        palette='coolwarm'
    )
    plt.xlim(smoking[col].min() - 0.3, smoking[col].max() + 0.1)
    plt.title(f'{col} by Gender & Smoking')
    plt.yticks([0,1], ['Female','Male'])

    handles, labels = plt.gca().get_legend_handles_labels()
    custom_labels = ['Non-smoker', 'Smoker']  # Customize labels
    plt.legend(handles, custom_labels, title='Smoking Status', loc='upper right')

    ax2 = plt.gca()
    ax2.set_facecolor("snow")

plt.tight_layout()
plt.show()
```

::: {.output .display_data}
![](008c48aac6890a6b30477fab937b4bf11b817e15.png)
:::
::::

::: {#97a25b2bec54d485 .cell .markdown}
**Results**

- Hemoglobin

  - Noticeably higher for smokers than non-smokers. This could be
    statistically significant since the violin plots for age do not
    indicate an increase in hemoglobin for people in middle age.
  - The right tail for smokers was also slightly fatter, indicating a
    greater distribution of higher hemoglobin.

- Weight and waist, though at first not much stood out, there was a
  similar pattern with the right tails being slightly thicker for non
  smokers, indicating smokers are more likely to be very heavy compared
  to non smokers, and more likely to have larger waists. This could be
  for other lifestyle reasons though, since smokers are more likely to
  have depression and more likely to be lower income.

- Serum Creatine had the same distribution for male smokers and non
  smokers. The serum creatine for women smokers had greater varience,
  though this could be due to small sample size.

- AST and ALT appeared unaffected

- It appears that female smokers have slightly less ldl, though this is
  likely due to other factors, since male smokers had about the same ldl
  as male non-smokers.

- There is a similar pattern on display with cholesterol, with male
  smokers having slightly elevated levels compared to male non-smokers,
  and female smokers having slightly lower levels.
:::

::: {#e0cb10d9fc99bc64 .cell .markdown}
# Boxplots of Features
:::

::::: {#4b6e5c3e454c68a .cell .code execution_count="22" ExecuteTime="{\"end_time\":\"2025-04-27T23:45:20.703383Z\",\"start_time\":\"2025-04-27T23:45:19.345311Z\"}"}
``` python
plt.figure(figsize=(10,10))

sns.boxenplot(data=smoking.drop(columns=['gender','dental_caries','hearing_left','hearing_right','eyesight_left','eyesight_right','tartar','smoking']), orient='h', palette='coolwarm')
plt.xscale('log')
plt.title('Box Plot of Numerical Features')
```

::: {.output .execute_result execution_count="22"}
    Text(0.5, 1.0, 'Box Plot of Numerical Features')
:::

::: {.output .display_data}
![](396227c8a0aea1af408364371afaf1e9af590e72.png)
:::
:::::

::: {#bb3e99b38f4d2e5a .cell .markdown}
The boxplots show numerous outliars and that the data differs
drastically in scale. Random forest is a great algorithm candidate for
these reasons along with the large sample size for our classification
problem.

# Model

## Model Selection

Random forest relies on a bootstrap method for obtaining samples, that
is it randomly selects data points with replacement.

Random forest is:

- Ideal for large samples
- Resistent to outliers
- Not sensitive to scale

We still have to deal with the fact that our dataset is highly
unbalanced both between genders and smoking vs non-smoking. Sofexley
uses stratified sampling, explaining that this is to ensure a roughly
even number of smokers and non smokers in each sample. I am going to
take their lead.

## Split Dataset
:::

::: {#6250545f28c8891b .cell .code execution_count="23" ExecuteTime="{\"end_time\":\"2025-04-27T23:45:25.730359Z\",\"start_time\":\"2025-04-27T23:45:25.726810Z\"}"}
``` python
def split_dataset(X, y, test_size=0.2, valid_size=0.3, seed=SEED):
    '''
    Returns the training, validation, test set pairs generated by stratified splitting.
    '''
    # Train_val & test split
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(strat_split.split(X, y))

    X_train_val, y_train_val = X.iloc[train_val_idx], y.iloc[train_val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # Train and val split
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=valid_size, random_state=seed)
    train_idx, val_idx = next(strat_split.split(X_train_val, y_train_val))

    X_train, y_train = X_train_val.iloc[train_idx], y_train_val.iloc[train_idx]
    X_val, y_val = X_train_val.iloc[val_idx], y_train_val.iloc[val_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test
```
:::

::: {#f0dc4262da8a654c .cell .code execution_count="24" ExecuteTime="{\"end_time\":\"2025-04-27T23:45:28.251683Z\",\"start_time\":\"2025-04-27T23:45:28.090245Z\"}"}
``` python
Smoking = smoking.reset_index().drop('ID', axis=1)

X = smoking.drop('smoking', axis = 1).copy()
y = pd.DataFrame(smoking['smoking'], index = smoking.index)

X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y, test_size=0.2,valid_size=0.3,seed=SEED)

y_train = y_train.values.ravel()
y_val = y_val.values.ravel()
y_test = y_test.values.ravel()
```
:::

:::: {#d0bb37aa5a51828e .cell .code ExecuteTime="{\"end_time\":\"2025-04-27T23:45:30.185440Z\",\"start_time\":\"2025-04-27T23:45:30.182856Z\"}"}
``` python
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)
```

::: {.output .stream .stdout}
    (31187, 24) (31187,)
    (13366, 24) (13366,)
    (11139, 24) (11139,)
:::
::::

::: {#7aa2dbfb4168f1b1 .cell .markdown}
# Model {#model}

Sofexley removed outliars using isolation forest, which I didnt fully
understand so I am not incorporating it.

I am however following these steps:

- Baseline model
- testing the baseline model:
  - with and without highly correlated features
  - feature importance: number of features
- Hyperparameter tuning

## Model Evaluation

I liked how Sofexley created functions for model evaluation. I thought
it kept things nice and clean.
:::

::: {#73f8d72fd036beff .cell .code execution_count="26" ExecuteTime="{\"end_time\":\"2025-04-28T00:00:04.463824Z\",\"start_time\":\"2025-04-28T00:00:04.460740Z\"}"}
``` python
def model_evaluation(train, predict):
    print("\nClassification Report")
    print(classification_report(train, predict))

    print("Confusion Matrix training")
    cm = confusion_matrix(train, predict)

    # Plot the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix", fontsize=16)
    plt.show()
```
:::

::: {#4763a0c30cb6ef82 .cell .markdown}
## Baseline Model
:::

::: {#f499c45735a7b57 .cell .code execution_count="27" ExecuteTime="{\"end_time\":\"2025-04-28T00:00:10.182230Z\",\"start_time\":\"2025-04-28T00:00:06.850799Z\"}"}
``` python
model = RandomForestClassifier(n_estimators = 100, random_state = SEED)

model.fit(X_train,y_train)

y_pred_tr = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_te = model.predict(X_test)
```
:::

::::: {#c39d7e2d5f08126f .cell .code execution_count="28" ExecuteTime="{\"end_time\":\"2025-04-28T00:00:12.782636Z\",\"start_time\":\"2025-04-28T00:00:12.749922Z\"}"}
``` python
print("Baseline Model-Training Dataset")
model_evaluation(y_train,y_pred_tr)
```

::: {.output .stream .stdout}
    Baseline Model-Training Dataset

    Classification Report
                  precision    recall  f1-score   support

               0       1.00      1.00      1.00     19732
               1       1.00      1.00      1.00     11455

        accuracy                           1.00     31187
       macro avg       1.00      1.00      1.00     31187
    weighted avg       1.00      1.00      1.00     31187

    Confusion Matrix training
:::

::: {.output .display_data}
![](b30fec4f6bd1a22c266fd525cd5571bf52d47ff3.png)
:::
:::::

::::: {#3003024b42b0384 .cell .code execution_count="29" ExecuteTime="{\"end_time\":\"2025-04-28T00:00:16.177005Z\",\"start_time\":\"2025-04-28T00:00:16.146542Z\"}"}
``` python
print("Baseline Model-Training Dataset")
model_evaluation(y_val,y_pred_val)
```

::: {.output .stream .stdout}
    Baseline Model-Training Dataset

    Classification Report
                  precision    recall  f1-score   support

               0       0.86      0.84      0.85      8457
               1       0.74      0.76      0.75      4909

        accuracy                           0.81     13366
       macro avg       0.80      0.80      0.80     13366
    weighted avg       0.81      0.81      0.81     13366

    Confusion Matrix training
:::

::: {.output .display_data}
![](ef5192cb9d751aad24291fc4b8ec5ee363ca22dc.png)
:::
:::::

::: {#ab61145c2310a0cf .cell .markdown}
### Comments {#comments}

As expected, the model is overfitting. We will see what happens as we
explore.
:::

::: {#5de8aed8b12f4404 .cell .markdown}
## Highly Correlated Features
:::

::::: {#d3060ac4e52200ae .cell .code execution_count="30" ExecuteTime="{\"end_time\":\"2025-04-28T00:00:19.513750Z\",\"start_time\":\"2025-04-28T00:00:19.139368Z\"}"}
``` python
corr_df = smoking.corr().abs() # creates a df of all the correlations, as a positive value between 0 and 1
mask = np.triu(np.ones_like(corr_df, dtype=bool)) # creates a template that is the upper triangle of a matrix of ones
tri_df = corr_df.mask(mask) # apply the mask to the data frame

plt.figure(figsize=(12,9))
sns.heatmap(tri_df, annot=True, cmap='cubehelix', fmt='.2f',)
plt.title('Positive Correlation Matrix')
```

::: {.output .execute_result execution_count="30"}
    Text(0.5, 1.0, 'Positive Correlation Matrix')
:::

::: {.output .display_data}
![](248a9a044e585a4988f3198e648c1a6ee0bb78bc.png)
:::
:::::

::::::::::::::::::::::: {#f7d2d4b2fb13a24e .cell .code execution_count="31" ExecuteTime="{\"end_time\":\"2025-04-28T00:00:38.975623Z\",\"start_time\":\"2025-04-28T00:00:22.347457Z\"}"}
``` python
# Find the columns that meet threshold.
thresholds = [0.5,0.6,0.7,0.75,0.85]
results={}
for threshold in thresholds:
    high_corr_features = [col for col in tri_df.columns if any(tri_df[col] > threshold)]

    X_train_corr = X_train.drop(high_corr_features, axis=1)
    X_val_corr = X_val.drop(high_corr_features, axis=1)

    model_corr = RandomForestClassifier(n_estimators=100, random_state=SEED)
    model_corr.fit(X_train_corr, y_train)

# Predict
    y_pred_tr_corr = model_corr.predict(X_train_corr)
    y_pred_val_corr = model_corr.predict(X_val_corr)

    print(f"Training Eval Report\n with correlation threshold:{threshold}")
    model_evaluation(y_train, y_pred_tr_corr)

    print(f"Validation Eval Report")
    model_evaluation(y_val, y_pred_val_corr)


```

::: {.output .stream .stdout}
    Training Eval Report
     with correlation threshold:0.5

    Classification Report
                  precision    recall  f1-score   support

               0       1.00      1.00      1.00     19732
               1       1.00      1.00      1.00     11455

        accuracy                           1.00     31187
       macro avg       1.00      1.00      1.00     31187
    weighted avg       1.00      1.00      1.00     31187

    Confusion Matrix training
:::

::: {.output .display_data}
![](892e80b8579e101b0956cc982d0b805c5c6c9781.png)
:::

::: {.output .stream .stdout}
    Validation Eval Report

    Classification Report
                  precision    recall  f1-score   support

               0       0.83      0.86      0.84      8457
               1       0.74      0.69      0.71      4909

        accuracy                           0.80     13366
       macro avg       0.78      0.78      0.78     13366
    weighted avg       0.80      0.80      0.80     13366

    Confusion Matrix training
:::

::: {.output .display_data}
![](d7873e0e27c47194db7aa67589d16aa8460546c7.png)
:::

::: {.output .stream .stdout}
    Training Eval Report
     with correlation threshold:0.6

    Classification Report
                  precision    recall  f1-score   support

               0       1.00      1.00      1.00     19732
               1       1.00      1.00      1.00     11455

        accuracy                           1.00     31187
       macro avg       1.00      1.00      1.00     31187
    weighted avg       1.00      1.00      1.00     31187

    Confusion Matrix training
:::

::: {.output .display_data}
![](a5232feb47f7016725618a1e044daa423c5fdcd5.png)
:::

::: {.output .stream .stdout}
    Validation Eval Report

    Classification Report
                  precision    recall  f1-score   support

               0       0.83      0.86      0.84      8457
               1       0.74      0.69      0.72      4909

        accuracy                           0.80     13366
       macro avg       0.78      0.78      0.78     13366
    weighted avg       0.80      0.80      0.80     13366

    Confusion Matrix training
:::

::: {.output .display_data}
![](3a999c8a0ae5705244f0c3746983a6efc00f29b8.png)
:::

::: {.output .stream .stdout}
    Training Eval Report
     with correlation threshold:0.7

    Classification Report
                  precision    recall  f1-score   support

               0       1.00      1.00      1.00     19732
               1       1.00      1.00      1.00     11455

        accuracy                           1.00     31187
       macro avg       1.00      1.00      1.00     31187
    weighted avg       1.00      1.00      1.00     31187

    Confusion Matrix training
:::

::: {.output .display_data}
![](a5232feb47f7016725618a1e044daa423c5fdcd5.png)
:::

::: {.output .stream .stdout}
    Validation Eval Report

    Classification Report
                  precision    recall  f1-score   support

               0       0.84      0.85      0.85      8457
               1       0.74      0.72      0.73      4909

        accuracy                           0.81     13366
       macro avg       0.79      0.79      0.79     13366
    weighted avg       0.80      0.81      0.81     13366

    Confusion Matrix training
:::

::: {.output .display_data}
![](0f531ef4364f63a48100e6fdb085b8753c41e52b.png)
:::

::: {.output .stream .stdout}
    Training Eval Report
     with correlation threshold:0.75

    Classification Report
                  precision    recall  f1-score   support

               0       1.00      1.00      1.00     19732
               1       1.00      1.00      1.00     11455

        accuracy                           1.00     31187
       macro avg       1.00      1.00      1.00     31187
    weighted avg       1.00      1.00      1.00     31187

    Confusion Matrix training
:::

::: {.output .display_data}
![](a5232feb47f7016725618a1e044daa423c5fdcd5.png)
:::

::: {.output .stream .stdout}
    Validation Eval Report

    Classification Report
                  precision    recall  f1-score   support

               0       0.86      0.84      0.85      8457
               1       0.74      0.76      0.75      4909

        accuracy                           0.81     13366
       macro avg       0.80      0.80      0.80     13366
    weighted avg       0.81      0.81      0.81     13366

    Confusion Matrix training
:::

::: {.output .display_data}
![](6b9ee249c71e62962db0f5d70ef69d6f1106a5ac.png)
:::

::: {.output .stream .stdout}
    Training Eval Report
     with correlation threshold:0.85

    Classification Report
                  precision    recall  f1-score   support

               0       1.00      1.00      1.00     19732
               1       1.00      1.00      1.00     11455

        accuracy                           1.00     31187
       macro avg       1.00      1.00      1.00     31187
    weighted avg       1.00      1.00      1.00     31187

    Confusion Matrix training
:::

::: {.output .display_data}
![](b30fec4f6bd1a22c266fd525cd5571bf52d47ff3.png)
:::

::: {.output .stream .stdout}
    Validation Eval Report

    Classification Report
                  precision    recall  f1-score   support

               0       0.86      0.84      0.85      8457
               1       0.74      0.76      0.75      4909

        accuracy                           0.81     13366
       macro avg       0.80      0.80      0.80     13366
    weighted avg       0.81      0.81      0.81     13366

    Confusion Matrix training
:::

::: {.output .display_data}
![](ef5192cb9d751aad24291fc4b8ec5ee363ca22dc.png)
:::
:::::::::::::::::::::::

::: {#6010a5a3eba73cfd .cell .markdown}
Since my model was still overfitting, I experimented with a few
thresholds for correlation. This was a dead end. The models all
overfitted on training data, and none of the validation models did
better than the baseline model.
:::

::: {#86344eff95827c58 .cell .markdown}
# Exploring Feature Importance
:::

::::: {#20ae0b66bd545d00 .cell .code execution_count="33" ExecuteTime="{\"end_time\":\"2025-04-28T00:00:59.657874Z\",\"start_time\":\"2025-04-28T00:00:43.481935Z\"}"}
``` python
clf = ExtraTreesClassifier(n_estimators=1000, random_state=SEED)
clf.fit(X_train.values, y_train)
# Extract feature importances from the ET model and plot them.
imp_features = pd.Series(clf.feature_importances_,
                         index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(7, 6))
sns.barplot(x=imp_features, y=imp_features.index, palette='winter')
plt.title('Feature Importances', fontsize=14)
plt.xlabel('Importance Score')
```

::: {.output .execute_result execution_count="33"}
    Text(0.5, 0, 'Importance Score')
:::

::: {.output .display_data}
![](19114911d9612418442c81489730d1ad8b8275f2.png)
:::
:::::

::: {#3c8edaadf3f09bd4 .cell .markdown}
Top 5 features:

- Gender
- Hemoglobin
- Height
- GTP
- Triglycerides

Gender and hemoglobin were expected as top contributers, as was height.
I looked into GTP and smoking online, and could not find information. I
could however find information that high triglyceride levels were
related to smoking.

Though I did not expect urine protein or tartar to have a large impact
on the model, I am surprised they have less of an impact than eyesight.
For that matter, I am equally surprised eyesight has the impact that it
does. I suspect this could be because smokers overall lead a less
healthy lifestyle, which could impact their eyes, or some other
unaccounted for demographic reason.

**Removed the model section.** After going back and trying several
thresholds of correlation to drop in order to reduce complexity, I
realized removing features was not helping.
:::

::: {#f1452c14f8200754 .cell .markdown}
# Model Hypertuning
:::

:::: {#3055ec279bef585f .cell .code execution_count="48" ExecuteTime="{\"end_time\":\"2025-04-28T02:39:48.402297Z\",\"start_time\":\"2025-04-28T02:36:41.209906Z\"}"}
``` python
param_grid = {
    'max_depth': [1, 5, 10, 20, 45],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [2, 4, 6, 8],
    'n_estimators': [10, 50, 100, 300, 500, 1000]
}
rf = RandomForestClassifier(random_state=SEED)

X_train_feat10 = X_train[imp_features.index[:10]]
X_val_feat10 = X_val[imp_features.index[:10]]

grid_search1 = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid_search1.fit(X_val_feat10, y_val)

print("Best Parameters:", grid_search1.best_params_)
print("Best Cross-Validation Score:", grid_search1.best_score_)
```

::: {.output .stream .stdout}
    Fitting 5 folds for each of 480 candidates, totalling 2400 fits
    Best Parameters: {'max_depth': 45, 'min_samples_leaf': 2, 'min_samples_split': 8, 'n_estimators': 1000}
    Best Cross-Validation Score: 0.7638779551519516
:::
::::

::::: {#8619bc6aa365bb4a .cell .code execution_count="56" ExecuteTime="{\"end_time\":\"2025-04-28T02:44:30.256019Z\",\"start_time\":\"2025-04-28T02:44:29.413742Z\"}"}
``` python
best_model = grid_search1.best_estimator_
y_test_pred_best=best_model.predict(X_test[imp_features.index[:10]])
model_evaluation(y_test,y_test_pred_best)
```

::: {.output .stream .stdout}

    Classification Report
                  precision    recall  f1-score   support

               0       0.83      0.80      0.81      7048
               1       0.67      0.71      0.69      4091

        accuracy                           0.77     11139
       macro avg       0.75      0.75      0.75     11139
    weighted avg       0.77      0.77      0.77     11139

    Confusion Matrix training
:::

::: {.output .display_data}
![](84fc8f94542a2073520e1478707d486d87186730.png)
:::
:::::

:::: {#1f44f67ca61411e8 .cell .code execution_count="43" ExecuteTime="{\"end_time\":\"2025-04-28T01:14:21.131322Z\",\"start_time\":\"2025-04-28T01:10:43.247420Z\"}"}
``` python
param_grid = {
    'max_depth': [1, 5, 10, 20, 45],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [2, 4, 6, 8],
    'n_estimators': [10, 50, 100, 300, 500, 1000]
}
rf = RandomForestClassifier(random_state=SEED)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_val, y_val)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)
```

::: {.output .stream .stdout}
    Fitting 5 folds for each of 480 candidates, totalling 2400 fits
    Best Parameters: {'max_depth': 45, 'min_samples_leaf': 2, 'min_samples_split': 6, 'n_estimators': 500}
    Best Cross-Validation Score: 0.7753254028414005
:::
::::

::::: {#5763c124f0dce1d7 .cell .code execution_count="58" ExecuteTime="{\"end_time\":\"2025-04-28T02:44:53.443073Z\",\"start_time\":\"2025-04-28T02:44:52.990132Z\"}"}
``` python
best_model2 = grid_search.best_estimator_
y_test_pred_best=best_model2.predict(X_test)
model_evaluation(y_test,y_test_pred_best)
```

::: {.output .stream .stdout}

    Classification Report
                  precision    recall  f1-score   support

               0       0.84      0.80      0.82      7048
               1       0.69      0.74      0.71      4091

        accuracy                           0.78     11139
       macro avg       0.76      0.77      0.77     11139
    weighted avg       0.78      0.78      0.78     11139

    Confusion Matrix training
:::

::: {.output .display_data}
![](24a3573e5039726b634da611b5bdf784cce9e6b6.png)
:::
:::::

::: {#875ecbf0d43d93dd .cell .markdown}
:::

:::: {#256310e33fbf3091 .cell .code execution_count="60" ExecuteTime="{\"end_time\":\"2025-04-28T02:56:36.395048Z\",\"start_time\":\"2025-04-28T02:54:34.631457Z\"}"}
``` python
print(f" Accuracy of best fit model all features:{cross_val_score(best_model2, X_test,y_test, cv=cv, scoring='accuracy').mean().round(4)*100}")

print(f" Accuracy of best fit model top 10 features:{cross_val_score(best_model, X_test,y_test, cv=cv, scoring='accuracy').mean().round(4)*100}")
```

::: {.output .stream .stdout}
     Accuracy of best fit model all features:77.22
     Accuracy of best fit model top 10 features:76.99000000000001
:::
::::
