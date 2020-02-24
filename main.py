# %%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009
wine = pd.read_csv('dataset/wine1/winequality-red.csv')
# zistovanie kvality


# https://archive.ics.uci.edu/ml/datasets/Wine+Quality?fbclid=IwAR19qa6Gcg2iOFSfntw3nCKgZ2bWDKvdkZLchpIOk7Fd0radtcylWqtTMHw
wine_red = pd.read_csv('dataset/wine2/winequality-red.csv')
wine_white = pd.read_csv('dataset/wine2/winequality-white.csv')
# zistovanie kvality


# https://www.kaggle.com/andytran11996/citibike-dataset-2017
citibike = pd.read_csv('dataset/citibike/citibike.csv')
# klasifikacia pohlavia

# https://www.kaggle.com/ronitf/heart-disease-uci
heart = pd.read_csv('dataset/heart/heart.csv')
# identifikacia infarktu

# https://www.kaggle.com/uciml/mushroom-classification
mushrooms = pd.read_csv('dataset/mushrooms/mushrooms.csv')
mushrooms.rename(columns={'class': 'eat'}, inplace=True)
# zistovanie ci je alebo nie je mozne jest hubu

# https://www.kaggle.com/iabhishekofficial/mobile-price-classification#train.csv
train_mobile = pd.read_csv('dataset/mobile/train.csv')
tast_mobile = pd.read_csv('dataset/mobile/test.csv')
mobile = pd.concat([train_mobile, tast_mobile])
# clasifikacia ceny

# %%


def printBoxPlot(x_value, y_value, df):
    fig = plt.figure(figsize=(10, 6))
    plt.xlabel(x_value)
    plt.ylabel(y_value)
    sns.barplot(x=x_value, y=y_value, data=df)
    pass


def printHistogram(df_with_col):
    df_with_col.plot(kind='hist', bins=50, figsize=(12, 12))
    plt.show()
    pass


def printCorrHeatMap(df):
    f, ax = plt.subplots(figsize=(18, 18))
    sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
    plt.show()
    pass


def printCorrHeatMapOneValue(df, value):
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(df.corr()[[value]].sort_values(value).tail(10),
                vmax=1, vmin=-1, annot=True, ax=ax)
    ax.invert_yaxis()


def printScarrlet(x_value, y_value, df):
    df.plot(kind='scatter', x=x_value,
            y=y_value, alpha=0.5, color='red')
    plt.xlabel(x_value)
    plt.ylabel(y_value)


def printInfoData(name_dataset, df, df_with_col, primary_col):
    print(name_dataset)
    print(df.shape)
    print("")
    print(df.head())
    print("")
    df.info()
    print("")
    printHistogram(df_with_col)
    printCorrHeatMap(df)
    printCorrHeatMapOneValue(df, primary_col)
    for col in df.columns:
        if col != primary_col:
            printBoxPlot(primary_col, col, df)
            printScarrlet(primary_col, col, df)
    pass


# %%
printInfoData("White Wine", wine_white, wine_white.quality, "quality")

# %%
printInfoData("Red Wine", wine_red, wine_red.quality, "quality")

# %%
printInfoData("Wine", wine, wine.quality, "quality")

# %%
printInfoData("Citibike", citibike, citibike.Gender, "Gender")


# %%
printInfoData("Heart", heart, heart.thal, "thal")

# %%
printInfoData("Mushrooms", mushrooms, mushrooms.eat, "eat")

# %%
printInfoData("Mobile", mobile, mobile.price_range, "price_range")


# %%
# print("wine 1")

# print(wine.shape)
# wine.head()
# wine.info()
# %%
# print("wine 2\n")
# print(wine_white.shape)
# print("")
# print(wine_white.head())
# print("")
# wine_white.info()
# print("")
# printHistogram(wine_white.quality)
# printCorrHeatMap(wine_white)
# printCorrHeatMapOneValue( wine_white, "quality" )
# for col in wine.columns:
#     if col != "quality":
#         printBoxPlot("quality", col, wine_white)
#         printScarrlet("quality", col, wine_white)
