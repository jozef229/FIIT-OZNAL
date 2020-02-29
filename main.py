# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import fbeta_score
from scipy import stats
from sklearn import datasets, metrics
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (RBF, ConstantKernel,
                                              ExpSineSquared,
                                              RationalQuadratic)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# %%

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
# printInfoData("Mobile", train_mobile, mobile.price_range, "price_range")


# %%

def addCelMoreOrLessMean(df, col):
    df[col + ">Mean"] = df.apply(lambda row: 1 if row[col]
                                 > df[col].mean() else 0, axis=1)


def addCelMoreOrLessMedian(df, col):
    df[col + ">Median"] = df.apply(lambda row: 1 if row[col]
                                   > df[col].median() else 0, axis=1)


def addCelMoreOrLessQuantile25(df, col):
    df[col + ">Quantile25"] = df.apply(lambda row: 1 if row[col]
                                       > df[col].quantile(0.25) else 0, axis=1)


def addCelMoreOrLessQuantile75(df, col):
    df[col + ">Quantile75"] = df.apply(lambda row: 1 if row[col]
                                       > df[col].quantile(0.75) else 0, axis=1)


def addCol(df, col):
    addCelMoreOrLessMean(df, col)
    addCelMoreOrLessMedian(df, col)
    addCelMoreOrLessQuantile25(df, col)
    addCelMoreOrLessQuantile75(df, col)


# %%


def iqr_outliers(dataset, bottom_quantile=0.25, top_quantile=0.75):
    Q1 = dataset.quantile(bottom_quantile)
    Q3 = dataset.quantile(top_quantile)
    IQR = Q3 - Q1
    dataset_out = dataset[
        ~((dataset < (Q1 - 1.5 * IQR)) | (dataset > (Q3 + 1.5 * IQR))).any(axis=1)
    ]
    return dataset_out


def z_score_outliers(dataset, threshold=3):
    dataset_out = dataset[
        (np.abs(stats.zscore(dataset.select_dtypes(exclude="object"))) < threshold).all(
            axis=1
        )
    ]
    return dataset_out


# %%

def CorrelationMatrixSelectFeatures(dataset, size_of_delet_corelation=0.95):
    corr_matrix = dataset.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    select_column = [
        column
        for column in upper.columns
        if any(upper[column] > size_of_delet_corelation)
    ]
    dataset_out = dataset.drop(dataset[select_column], axis=1)
    return dataset_out


def vifSelectFeatures(dataset, thresh=100.0):
    variables = list(range(dataset.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [
            variance_inflation_factor(dataset.iloc[:, variables].values, ix)
            for ix in range(dataset.iloc[:, variables].shape[1])
        ]
        maxloc = vif.index(max(vif))

        if max(vif) > thresh:
            del variables[maxloc]
            dropped = True
    dataset_out = dataset.iloc[:, variables]
    return dataset_out


def selectKBest(dataset, X_data, y_data):
    dataset_out = SelectKBest(chi2, k=2).fit_transform(X_data, y_data)
    return dataset_out

# %%


classifiers = [
    AdaBoostClassifier(),
    DecisionTreeClassifier(max_depth=5),
    ExtraTreesClassifier(n_estimators=5, criterion="entropy", max_features=2),
    GaussianNB(),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    KNeighborsClassifier(),
    LinearDiscriminantAnalysis(),
    LogisticRegression(
        penalty="l1", dual=False, max_iter=110, solver="liblinear", multi_class="auto"
    ),
    MLPClassifier(alpha=1, max_iter=1000),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    SVC(kernel="poly", C=0.025),
    SVC(kernel="linear", C=0.025),
    SVC(kernel="sigmoid", gamma=2),
    SVC(kernel="rbf", gamma=2, C=1),
    QuadraticDiscriminantAnalysis(),
]


classifiersNames = [
    "AdaBoost",
    "Decision Tree",
    "Extra Trees",
    "Naive Bayes",
    "Gaussian Process",
    "Nearest Neighbors",
    "Linear Discriminant Analysis",
    "Logistic Regression",
    "Neural Net",
    "Random Forest",
    "SVM Poly",
    "SVM Sigmoid",
    "SVM Linear ",
    "SVM RBF",
    "QDA",
]


outliersName = [
    "iqr",
    "z_score",
]

featureSelectionName = [
    "cor",
    "vif"
]


# %%


def printFinalTable(df_print, nameOutlier, nameFeatureSelection, model, nameIndex, train, test):
    df_print = df_print.append({"Outliers": outliersName[nameOutlier],
                                "Feature Selection": featureSelectionName[nameFeatureSelection],
                                "Model": classifiersNames[nameIndex],
                                "ACC": accuracy_score(train, test),
                                "Recall macro": recall_score(train, test, average="macro"),
                                "Recall micro": recall_score(train, test, average="micro"),
                                "Recall weighted": recall_score(train, test, average="weighted"),
                                "F1 macro": f1_score(train, test, average="macro", labels=np.unique(test)),
                                "F1 micro": f1_score(train, test, average="micro", labels=np.unique(test)),
                                "F1 weighted": f1_score(train, test, average="weighted", labels=np.unique(test))}, ignore_index=True)
    print(df_print.head())


def classificationModel(typeModel, X, y):
    model = classifiers[typeModel]
    model.fit(X, y)
    return model


# %%


train_mobile = pd.read_csv('dataset/mobile/train.csv')
main_value = 'price_range'
train_mobile_without_price = train_mobile.copy()
del train_mobile_without_price[main_value]
# y_data = train_mobile[main_value]

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


other_value = train_mobile_without_price.columns.tolist()
X_data = train_mobile[other_value]
y_data = train_mobile[main_value]

train, test = train_test_split(df, test_size=0.2)


X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, shuffle=False
)

df_stats_model = pd.DataFrame()

for i in range(len(outliersName)):
    df_mobile = X_train.copy()
    if i == 0:
        df_mobile = iqr_outliers(df_mobile)
    if i == 1:
        z_score_outliers(df_mobile)
    for u in range(len(featureSelectionName)):

        if u == 0:
            df_mobile = CorrelationMatrixSelectFeatures(df_mobile)
        if u == 1:
            df_mobile = vifSelectFeatures(df_mobile)

        for i in range(len(classifiers)):
            model = classificationModel(i, X_train, y_train)
            X_test_predict = model.predict(X_test)
            printFinalTable(df_stats_model, i, u, model,
                            i, X_test_predict, y_test)


df_stats_model.to_csv(r'Dataset/columns_stats_model.csv',
                      index=False, header=True)


# print("standard = ", train_mobile.shape)
# train_mobil_iqr = iqr_outliers(train_mobile)
# print("iqr = ", train_mobil_iqr.shape)

# train_mobil_z_score = z_score_outliers(train_mobile)
# print("z_score = ", train_mobil_z_score.shape)


# addCol(train_mobile, "battery_power")
# addCol(train_mobile, "clock_speed")
# addCol(train_mobile, "int_memory")
# addCol(train_mobile, "m_dep")
# addCol(train_mobile, "mobile_wt")
# addCol(train_mobile, "n_cores")
# addCol(train_mobile, "px_height")
# addCol(train_mobile, "px_width")
# addCol(train_mobile, "pc")
# addCol(train_mobile, "ram")
# addCol(train_mobile, "sc_h")
# addCol(train_mobile, "sc_w")
# addCol(train_mobile, "talk_time")


# print("standard = ", train_mobile.shape)
# train_mobil_cor = CorrelationMatrixSelectFeatures(train_mobile)
# print("cor = ", train_mobil_cor.shape)

# train_mobil_vif = vifSelectFeatures(train_mobile)
# print("vif = ", train_mobil_vif.shape)


# df = pd.read_csv(
#     "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
# )

# df = df.replace("virginica", 3)
# df = df.replace("versicolor", 2)
# df = df.replace("setosa", 1)
# df["species"] = df["species"].astype(int)

# other_value = df.columns.tolist()

# main_value = 'species'
# other_value = list(filter(lambda x: x != main_value, help))

# X_data = df[other_value]
# y_data = df[main_value]


# X_train, X_test, y_train, y_test = train_test_split(
#     X_data, y_data, test_size=0.2, shuffle=False
# )

# for i in range(len(classifiers)):
#     model = classificationModel(i, X_train, y_train)
#     X_test_predict = model.predict(X_test)
#     printClassificationReport(model, i, X_test_predict, y_test)


# %%
