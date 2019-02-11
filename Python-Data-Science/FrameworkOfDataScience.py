# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 11:10:30 2019

@author: Tung1108
@Reference: https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/bonus%20content/effective%20data%20visualization/Bonus%20-%20Effective%20Multi-dimensional%20Data%20Visualization.ipynb
"""
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns 

######################################################################################
############################ Load and merge datasets #################################
######################################################################################

white_wine = pd.read_csv('winequality-white.csv', sep=';')
red_wine = pd.read_csv('winequality-red.csv', sep=';') 

red_wine['wine_type'] = 'red'
white_wine['wine_type'] = 'white'

red_wine['quality_label'] = red_wine['quality'].apply(lambda value: 'low'
                                                        if value <= 5 else 'medium'
                                                        if value <= 7 else 'high')
red_wine['quality_label'] = pd.Categorical(red_wine['quality_label'],
                                            categories=['low', 'median', 'high'])
white_wine['quality_label'] = white_wine['quality'].apply(lambda value: 'low'
                                                          if value <= 5 else 'medium'
                                                          if value <= 7 else 'high')
white_wine['quality_label'] = pd.Categorical(white_wine['quality_label'],
                                              categories=['low','medium','high'])

# Merge datasets 
wines = pd.concat([red_wine, white_wine])

# Re-shuffle records just to randomize data points 
wines = wines.sample(frac=1, random_state=42).reset_index(drop=True)
wines.head()


######################################################################################
################# Exploratory Data Analysis and Visualization ########################
######################################################################################

# Descriptive Statistics

subset_attributes = ['residual sugar', 'total sulfur dioxide', 'sulphates', 'alcohol', 
                     'volatile acidity', 'quality']
rs = round(red_wine[subset_attributes].describe(),2)
ws = round(white_wine[subset_attributes].describe(),2)
rs_ws = pd.concat([rs,ws], axis=1, keys=['Red Wine Statistics', 'White Wine Statistics'])

subset_attributes_total = ['alcohol', 'volatile acidity', 'pH', 'quality']
ls = round(wines[wines['quality_label'] == 'low'][subset_attributes_total].describe(),2)
ms = round(wines[wines['quality_label'] == 'medium'][subset_attributes_total].describe(),2)
hs = round(wines[wines['quality_label'] == 'high'][subset_attributes_total].describe(),2)
ls_ms_hs = pd.concat([ls,ms,hs], axis=1, keys=['Low Quality Wine', 'Medium Quality Wine', 
                                                 'High Quality Wine'])


# Visualizing one dimension

wines.hist(bins=15, color='steelblue', edgecolor='black',linewidth=1.0,
           xlabelsize=8, ylabelsize=8, grid=False)
plt.tight_layout(rect=(0,0,1.2,1.2))

fig = plt.figure(figsize = (6,4))
title = fig.suptitle("Sulphates Content in Wine", fontsize = 14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1,1)
ax.set_xlabel("Sulphates")
ax.set_ylabel("Frequency")
ax.text(1.2, 800, r'$\mu$=' + str(round(wines['sulphates'].mean(),2)), fontsize=12)
freq, bins, patches = ax.hist(wines['sulphates'], color='steelblue', bins=15,
                              edgecolor='black', linewidth=1)

fig = plt.figure(figsize = (6,4))
title = fig.suptitle("Sulphates Content in Wine", fontsize = 14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel("Sulphates")
ax1.set_ylabel("Density")
sns.kdeplot(wines['sulphates'], ax=ax1, shade=True, color='steelblue')

# Visualizing two dimensions

f, ax = plt.subplots(figsize=(10,6))
corr = wines.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm", fmt='.2f',
                 linewidths = .05)
f.subplots_adjust(top=0.93)
t = f.suptitle('Wine Attributes Correlation Heatmap', fontsize = 14)

cols = ['density', 'residual sugar', 'total sulfur dioxide', 'fixed acidity']
pp = sns.pairplot(wines[cols], size=1.8, aspect=1.8, 
                  plot_kws=dict(edgecolor="k", linewidth = 0.5), 
                  diag_kind="kde", diag_kws=dict(shade=True))
fig = pp.fig
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Wine Attributes Pairwise Plots', fontsize=14)

from sklearn.preprocessing import StandardScaler
subset_df = wines[cols]
ss = StandardScaler()
scaled_df = ss.fit_transform(subset_df)
scaled_df = pd.DataFrame(scaled_df, columns=cols)
final_df = pd.concat([scaled_df, wines['wine_type']], axis=1)
final_df.head()

plt.scatter(wines['sulphates'], wines['alcohol'], 
            alpha=0.4, edgecolors='w')
plt.xlabel('Sulphates')
plt.ylabel('Alcohol')
plt.title('Wine Sulphates - Alcohol Content', y=1.05)
jp = sns.jointplot(x='sulphates', y='alcohol', data = wines, 
                   kind='reg', space=0, size=5, ratio=4)

cp = sns.countplot(x="quality", hue="wine_type", data=wines, 
                   palette={"red": "#FF9999", "white": "#FFE888"})

fig = plt.figure(figsize = (6,4))
title = fig.suptitle("Sulphates Content in Wine", fontsize = 14)
fig.subplots_adjust(top=0.85, wspace=0.3)
ax = fig.add_subplot(1,1,1)

































