# %%

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set display
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", -1)

# https://www.kaggle.com/iabhishekofficial/mobile-price-classification#train.csv
train_mobile = pd.read_csv('dataset/mobile/train.csv')
test_mobile = pd.read_csv('dataset/mobile/test.csv')
mobile = pd.concat([train_mobile, test_mobile])

# %%


def getInfo(columna_name):
    switcher = {
        "id": "ID",
        "battery_power": "Total energy a battery can store in one time measured in mAh",
        "blue": "Has bluetooth or not",
        "clock_speed": "speed at which microprocessor executes instructions",
        "dual_sim": "Has dual sim support or not",
        "fc": "Front Camera mega pixels",
        "four_g": "Has 4G or not",
        "int_memory": "Internal Memory in Gigabytes",
        "m_dep": "Mobile Depth in cm",
        "mobile_wt": "Weight of mobile phone",
        "n_cores": "Number of cores of processor",
        "pc": "Primary Camera mega pixels",
        "px_height": "Pixel Resolution Height",
        "px_width": "Pixel Resolution Width",
        "ram": "Random Access Memory in Megabytes",
        "sc_h": "Screen Height of mobile in cm",
        "sc_w": "Screen Width of mobile in cm",
        "talk_time": "longest time that a single battery charge will last when you are",
        "three_g": "Has 3G or not",
        "touch_screen": "Has touch screen or not",
        "wifi": "Has wifi or not",
        "price_range": "This is the target variable with value of 0(low cost), 1(medium cost), 2(high cost) and 3(very high cost).",
    }
    return switcher.get(columna_name, "nothing")


def printBoxPlot(x_value, y_value, df):
    fig = plt.figure(figsize=(10, 6))
    plt.xlabel(x_value)
    plt.ylabel(y_value)
    plt.title(x_value + ' and ' + y_value)
    sns.barplot(x=x_value, y=y_value, data=df)
    pass


def printBoxPlot2(x, y, df):
    sns.boxplot(data=df, x=x, y=y, notch=True)


def printHistogram(df):
    fig = plt.figure(figsize=(15, 20))
    ax = fig.gca()
    hist = df.hist(ax=ax)
    plt.savefig('dataset/histograms.png')


def printCorrHeatMap(df):
    f, ax = plt.subplots(figsize=(18, 18))
    sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
    plt.show()
    plt.title("Correlation map")
    plt.savefig('dataset/corrHeatMap.png')
    pass


def printCorrHeatMapOneValue(df, value):
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.title("Correlation by " + value)
    sns.heatmap(df.corr()[[value]].sort_values(value).tail(10),
                vmax=1, vmin=-1, annot=True, ax=ax)
    ax.invert_yaxis()
    plt.savefig('dataset/corrHeatMapPrice.png')


def printScarrlet(x_value, y_value, df):
    df.plot(kind='scatter', x=x_value,
            y=y_value, alpha=0.5, color='red')
    plt.xlabel(x_value)
    plt.ylabel(y_value)
    plt.title(x_value + ' and ' + y_value)


def printAllScarrlet(df):
    axs = pd.plotting.scatter_matrix(df[['battery_power', 'clock_speed', 'fc', 'int_memory', 'mobile_wt',
                                         'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'price_range']], figsize=(15, 15))
    plt.savefig('dataset/scarrletAll.png')


def printInfoData(name_dataset, df, df_with_col, primary_col):
    print(name_dataset)
    print(df.shape)
    print("")
    print(df.head())
    print("")
    df.info(verbose=True, null_counts=False)
    print("")
    printHistogram(df)
    printCorrHeatMap(df)
    printCorrHeatMapOneValue(df, primary_col)
    printAllScarrlet(df)
    df_stats = pd.DataFrame()
    df_type = pd.DataFrame()
    for col in df.columns:
        df_stats = df_stats.append({'Columns': col, "Mean": df[col].mean(), "Median": df[col].median(), "Variance": df[col].var(), "Standard_deviation": df[col].std(), "Max": df[col].max(
        ), "Min": df[col].min(), "Quantile_25%": df[col].quantile(0.25), "Quantile_50%": df[col].quantile(0.5), "Quantile_75%": df[col].quantile(0.75)}, ignore_index=True)
        df_type = df_type.append({'Columns': col, "Count": df[col].count(
        ), "Type": df[col].dtype, "Unique_value": df[col].nunique(), "Info": getInfo(col)}, ignore_index=True)
        if col != "quality":
            printBoxPlot("price_range", col, df)
            printBoxPlot2("price_range", col, df)

    df_stats.to_csv(r'Dataset/columns_stats.csv', index=False, header=True)
    df_type.to_csv(r'Dataset/columns_type.csv', index=False, header=True)


# %%
# printInfoData("Mobile", mobile, mobile.price_range, "price_range")
# printInfoData("Mobile", test_mobile, mobile.price_range, "price_range")
printInfoData("Mobile", train_mobile, mobile.price_range, "price_range")


# %%

def addCelMoreOrLessMean(df, col):
    df[col + ">Mean"] = df.apply(lambda row: 1 if row[col]
                                 > df[col].mean() else 0, axis=1)


def addCelMoreOrLessMedian(df, col):
    df[col + ">Median"] = df.apply(lambda row: 1 if row[col]
                                   > df[col].median() else 0, axis=1)


def addCelMoreOrLessStd(df, col):
    df[col + ">Std"] = df.apply(lambda row: 1 if row[col]
                                > df[col].std() else 0, axis=1)


def addCelMoreOrLessQuantile25(df, col):
    df[col + ">Quantile25"] = df.apply(lambda row: 1 if row[col]
                                       > df[col].quantile(0.25) else 0, axis=1)


def addCelMoreOrLessQuantile50(df, col):
    df[col + ">Quantile50"] = df.apply(lambda row: 1 if row[col]
                                       > df[col].quantile(0.50) else 0, axis=1)


def addCelMoreOrLessQuantile75(df, col):
    df[col + ">Quantile75"] = df.apply(lambda row: 1 if row[col]
                                       > df[col].quantile(0.75) else 0, axis=1)


def addCol(df, col):
    addCelMoreOrLessMean(df, col)
    addCelMoreOrLessMedian(df, col)
    addCelMoreOrLessStd(df, col)
    addCelMoreOrLessQuantile25(df, col)
    addCelMoreOrLessQuantile50(df, col)
    addCelMoreOrLessQuantile75(df, col)


addCol(train_mobile, "battery_power")
addCol(train_mobile, "clock_speed")
addCol(train_mobile, "int_memory")
addCol(train_mobile, "m_dep")
addCol(train_mobile, "mobile_wt")
addCol(train_mobile, "n_cores")
addCol(train_mobile, "px_height")
addCol(train_mobile, "px_width")
addCol(train_mobile, "pc")
addCol(train_mobile, "ram")
addCol(train_mobile, "sc_h")
addCol(train_mobile, "sc_w")
addCol(train_mobile, "talk_time")

print(train_mobile.head())
# %%


# %%
