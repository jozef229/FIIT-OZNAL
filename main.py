# %%

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from sklearn import datasets, metrics
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (RBF, ConstantKernel, DotProduct,
                                              ExpSineSquared, Matern,
                                              RationalQuadratic)
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, fbeta_score,
                             recall_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     train_test_split)
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

warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    sns_barplot = sns.barplot(x=x_value, y=y_value, data=df)
    fig = sns_barplot.get_figure()
    fig.savefig('Information/boxplot2/' + x_value + ' and ' + y_value + '.png')
    pass


def printBoxPlot2(x, y, df):
    sns_boxPlot = sns.boxplot(data=df, x=x, y=y, notch=True)
    fig = sns_boxPlot.get_figure()
    fig.savefig('Information/boxplot/' + x + ' and ' + y + '.png')


def printHistogram(df):
    fig = plt.figure(figsize=(15, 20))
    ax = fig.gca()
    hist = df.hist(ax=ax)
    plt.savefig('Information/histograms.png')
    plt.show(block=False)
    plt.close('all')


def printCorrHeatMap(df):
    f, ax = plt.subplots(figsize=(18, 18))
    sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
    plt.savefig('Information/corrHeatMap.png')
    plt.show(block=False)
    plt.close('all')
    pass


def printCorrHeatMapOneValue(df, value):
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.title("Correlation by " + value)
    sns.heatmap(df.corr()[[value]].sort_values(value).tail(10),
                vmax=1, vmin=-1, annot=True, ax=ax)
    ax.invert_yaxis()
    plt.savefig('Information/corrHeatMapPrice.png')
    plt.show(block=False)
    plt.close('all')


def printScarrlet(x_value, y_value, df):
    df.plot(kind='scatter', x=x_value,
            y=y_value, alpha=0.5, color='red')
    plt.xlabel(x_value)
    plt.ylabel(y_value)
    plt.title(x_value + ' and ' + y_value)
    plt.savefig('Information/scarrlet/' + x_value + ' and ' + y_value + '.png')
    plt.show(block=False)
    plt.close('all')


def printAllScarrlet(df):
    axs = pd.plotting.scatter_matrix(df[['battery_power', 'clock_speed', 'fc', 'int_memory', 'mobile_wt',
                                         'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'price_range']], figsize=(15, 15))
    plt.savefig('Information/scarrletAll.png')
    plt.close('all')


def iqr_dependence(df):
    bottom_quantile = 0.25
    top_quantile = 0.75
    Q1 = df.quantile(bottom_quantile)
    Q3 = df.quantile(top_quantile)
    IQR = Q3 - Q1
    print(IQR)
    help = (df < (Q1 - 1.5 * IQR)
            ) | (df > (Q3 + 1.5 * IQR))
    print(help.head())
    print()
    print(help.describe())
    print()

    iqr_data = help.describe()

    cm = sns.light_palette("green", as_cmap=True)
    styled = iqr_data.style.background_gradient(cmap=cm)
    styled.to_excel('Information/iqr_data.xlsx', engine='openpyxl')

    plt.boxplot(train_mobile.three_g)
    plt.savefig('Information/boxPlot_three_g.png')
    plt.close('all')

    plt.boxplot(train_mobile.fc)
    plt.savefig('Information/boxPlot_fc.png')
    plt.close('all')

    plt.boxplot(train_mobile.px_height)
    plt.savefig('Information/boxPlot_px_height.png')
    plt.close('all')


def z_score_dependence(df):
    train_mobile_z_score = df.copy()
    help = np.abs(stats.zscore(
        (train_mobile_z_score.select_dtypes(exclude="object"))) < 3).all(axis=1)

    np.count_nonzero(help == True)

    for col in train_mobile_z_score.columns:
        sns.distplot(train_mobile_z_score[col], color="maroon")
        plt.xlabel(col, labelpad=14)
        plt.ylabel("probability of occurence", labelpad=14)
        plt.title("Distribution of " + col, y=1.015, fontsize=20)
        plt.savefig('Information/z_score/' + "Distribution of " + col + '.png')
        plt.show(block=False)
        plt.close('all')

        train_mobile_z_score['us_z-score'] = (train_mobile_z_score[col] -
                                              train_mobile_z_score[col].mean())/train_mobile_z_score[col].std()

        train_mobile_z_score['us_z-score'].hist(color='slategray')
        plt.title("Standard Normal Distribution of " +
                  col, y=1.015, fontsize=22)
        plt.xlabel("z-score", labelpad=14)
        plt.ylabel("frequency", labelpad=14)
        plt.savefig('Information/z_score/' +
                    "Standard Normal Distribution of " + col + '.png')
        plt.show(block=False)
        plt.close('all')


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
    z_score_dependence(df)
    iqr_dependence(df)
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
    print("last")
    df_stats.to_csv(r'Information/columns_stats.csv', index=False, header=True)
    df_type.to_csv(r'Information/columns_type.csv', index=False, header=True)


# %%

train_mobile = pd.read_csv('dataset/mobile/train.csv')

# Print and save all information about train_model
# https://www.kaggle.com/iabhishekofficial/mobile-price-classification#train.csv
printInfoData("Mobile", train_mobile, train_mobile.price_range, "price_range")


# %%

corr_data_price = train_mobile[train_mobile.columns[1:]].corr()[
    'price_range'][:]

cm = sns.light_palette("green", as_cmap=True)
styled = corr_data_price.to_frame().style.background_gradient(cmap=cm)
styled.to_excel('Information/corr_data_price.xlsx', engine='openpyxl')

# %%
# help print
print(train_mobile.head())

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


def iqr_outliers(dataset, bottom_quantile=0.25, top_quantile=0.75):
    Q1 = dataset.quantile(bottom_quantile)
    Q3 = dataset.quantile(top_quantile)
    IQR = Q3 - Q1
    dataset_out = dataset[~((dataset < (Q1 - 1.5 * IQR))
                            | (dataset > (Q3 + 1.5 * IQR))).any(axis=1)]
    return dataset_out


def z_score_outliers(dataset, threshold=3):
    dataset_out = dataset[
        (np.abs(stats.zscore(dataset.select_dtypes(exclude="object"))) < threshold).all(
            axis=1
        )
    ]
    return dataset_out


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


classifiersParams = {
    "AdaBoost": {
        'n_estimators': range(40, 80, 6),
        'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
        'algorithm': ['SAMME', 'SAMME.R'],
    },
    "Decision Tree": {
        'min_samples_split': range(10, 400, 25),
        'max_features': ['sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7, 9, 10, 11, 12, 15, 30, 40, 70, 120, 150],
        'criterion': ['gini', 'entropy'],
    },
    "Extra Trees": {
        "n_estimators": range(10, 50, 4),
        "criterion": ("gini", "entropy"),
    },
    "Nearest Neighbors": {
        "n_neighbors": range(5, 9),
        "leaf_size": [1, 3, 5],
        "algorithm": ["auto", "kd_tree", "ball_tree", "brute"],
        "n_jobs": [-1],
    },
    "Logistic Regression": {
        "penalty": ["l1", "l2", "elasticnet"],
        "C": np.logspace(-5, 5, 15),
        "solver": ['saga'],
        "l1_ratio": [0, 1],
    },
    "Neural Net": {
        "solver": ["lbfgs", "sgd", "adam"],
        "max_iter": [1000, 1800],
        "alpha": 10.0 ** -np.arange(1, 4),
        "hidden_layer_sizes": np.arange(10, 13),
        "random_state": [0, 1, 3, 6],
    },
    "Random Forest": {
        "max_depth": range(20, 60, 6),
        "n_estimators": range(20, 60, 6),
        "max_features": ["sqrt", "log2"],
    },
    "SVM Sigmoid": {"kernel": ["sigmoid"], "degree": range(1, 4), "C": [1, 10]},
    "SVM Linear": {
        "kernel": ["linear"],
        "degree": range(1, 4),
        "C": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
    },
    "SVM RBF": {
        "kernel": ["rbf"],
        "degree": range(1, 4),
        "gamma": np.logspace(-4, 3, 10),
        "C": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
    },
}

classifiersParamsRandomSearch = {
    "AdaBoost": {
        'n_estimators': range(50, 200),
        'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
        'algorithm': ['SAMME', 'SAMME.R'],
    },
    "Decision Tree": {
        'min_samples_split': range(10, 500, 15),
        'max_features': ['sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150],
        'criterion': ['gini', 'entropy'],
    },
    "Extra Trees": {
        "n_estimators": range(10, 80),
        "criterion": ("gini", "entropy"),
    },
    "Nearest Neighbors": {
        "n_neighbors": range(3, 20),
        "leaf_size": [1, 3, 5, 7],
        "algorithm": ["auto", "kd_tree", "ball_tree", "brute"],
        "n_jobs": [-1],
    },
    "Logistic Regression": {
        "penalty": ["l1", "l2", "elasticnet"],
        "C": np.logspace(-5, 5),
        "solver": ['saga'],
        "l1_ratio": [0, 1],
    },
    "Neural Net": {
        "solver": ["lbfgs", "sgd", "adam"],
        "max_iter": [1000, 1400, 1800, 2000, 2500, 3000],
        "alpha": 10.0 ** -np.arange(1, 5),
        "hidden_layer_sizes": np.arange(10, 14),
        "random_state": [0, 1, 3, 6, 9],
    },
    "Random Forest": {
        "max_depth": range(20, 70),
        "n_estimators": range(10, 70),
        "max_features": ["sqrt", "log2", None],
    },
    "SVM Sigmoid": {"kernel": ["sigmoid"], "degree": range(1, 5), "C": [1, 10]},
    "SVM Linear": {
        "kernel": ["linear"],
        "degree": range(1, 4),
        "C": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
    },
    "SVM RBF": {
        "kernel": ["rbf"],
        "degree": range(1, 6),
        "gamma": np.logspace(-4, 3, 30),
        "C": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
    },
}


classifiersNames = [
    "AdaBoost",
    "Decision Tree",
    "Extra Trees",
    "Nearest Neighbors",
    "Logistic Regression",
    "Neural Net",
    "Random Forest",
    "SVM Sigmoid",
    "SVM Linear",
    "SVM RBF",
    "QDA",
    "Naive Bayes",
    "Linear Discriminant Analysis",
    "Gaussian Process",
]

classifiers = [
    AdaBoostClassifier(),
    DecisionTreeClassifier(max_depth=5),
    ExtraTreesClassifier(n_estimators=5, criterion="entropy", max_features=2),

    KNeighborsClassifier(),
    LogisticRegression(
        penalty="l1", dual=False, max_iter=110, solver="liblinear", multi_class="auto"
    ),
    MLPClassifier(alpha=1, max_iter=1000),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    SVC(kernel="sigmoid", gamma=2),
    SVC(kernel="linear", C=0.025),
    SVC(kernel="rbf", gamma=2, C=1),
    QuadraticDiscriminantAnalysis(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
]


outliersName = [
    "iqr",
    "z_score",
]

featureSelectionName = [
    "cor",
    "vif"
]

hyperparameterEstimation = [
    "nothing",
    "randomized search",
    "grid search"
]


def printFinalTable(df_print, records, features, nameOutlier, nameFeatureSelection, model, nameIndex, train, test, df_actual, hypEstimation):
    return df_print.append({
        "Model Params": model,
        "Records": records,
        "Hyperparameter estimation": hyperparameterEstimation[hypEstimation],
        "Features": features,
        "Outliers": outliersName[nameOutlier],
        "Outliers records": df_actual.shape[0],
        "Feature Selection": featureSelectionName[nameFeatureSelection],
        "Features with FS": df_actual.shape[1],
        "Model": classifiersNames[nameIndex],
        "ACC": accuracy_score(train, test),
        "Recall macro": recall_score(train, test, average="macro"),
        "Recall micro": recall_score(train, test, average="micro"),
        "Recall weighted": recall_score(train, test, average="weighted"),
        "F1 macro": f1_score(train, test, average="macro", labels=np.unique(test)),
        "F1 micro": f1_score(train, test, average="micro", labels=np.unique(test)),
        "F1 weighted": f1_score(train, test, average="weighted", labels=np.unique(test))}, ignore_index=True)


def classificationModel(validation, typeModel, X, y, heNumber, iteration=20):
    if heNumber == 0:
        model = classifiers[typeModel]
    if heNumber == 1:
        try:
            model = RandomizedSearchCV(
                classifiers[typeModel], param_distributions=classifiersParamsRandomSearch[classifiersNames[typeModel]], n_iter=iteration)
        except KeyError:
            validation = False
    if heNumber == 2:
        try:
            model = GridSearchCV(
                classifiers[typeModel], param_grid=classifiersParams[classifiersNames[typeModel]])
        except KeyError:
            validation = False
    model.fit(X, y)
    print("end hp")
    if heNumber == 0:
        return model
    else:
        return model.best_estimator_


def addMultiCol(df):
    addCol(df, "battery_power")
    addCol(df, "clock_speed")
    addCol(df, "int_memory")
    addCol(df, "m_dep")
    addCol(df, "mobile_wt")
    addCol(df, "n_cores")
    addCol(df, "px_height")
    addCol(df, "px_width")
    addCol(df, "pc")
    addCol(df, "ram")
    addCol(df, "sc_h")
    addCol(df, "sc_w")
    addCol(df, "talk_time")


# %%
train_mobile = pd.read_csv('dataset/mobile/train.csv')
main_value = 'price_range'

train, test = train_test_split(train_mobile, test_size=0.2)
y_test = test[main_value]
df_stats_model = pd.DataFrame()
initialProduct = train_mobile.shape[0] - 1
initialFeature = train_mobile.shape[1]
addMultiCol(test)

for outlier in range(len(outliersName)):
    df_mobile_out = train.copy()
    if outlier == 0:
        df_mobile_out = iqr_outliers(df_mobile_out)
    if outlier == 1:
        z_score_outliers(df_mobile_out)
    if df_mobile_out.shape[0] != 0:
        addMultiCol(df_mobile_out)
        print(df_mobile_out.head())
        print(df_mobile_out.columns.tolist())

        for selectFeature in range(len(featureSelectionName)):
            df_mobile = df_mobile_out.copy()
            if selectFeature == 0:
                df_mobile = CorrelationMatrixSelectFeatures(df_mobile)
            if selectFeature == 1:
                df_mobile = vifSelectFeatures(df_mobile)
            other_value = df_mobile.columns.tolist()
            y_train = df_mobile[main_value]
            X_train = df_mobile[list(
                filter(lambda x: x != main_value, other_value))]
            X_test = test[list(
                filter(lambda x: x != main_value, other_value))]
            for modelNumber in range(len(classifiers)):
                for hypEstimation in range(len(hyperparameterEstimation)):
                    start_time = time.time()
                    t = time.localtime()
                    current_time = time.strftime("%H:%M:%S", t)
                    print("start-", current_time)
                    print(str(df_mobile.shape[0]) +
                          "and" + str(df_mobile.shape[1]))
                    if modelNumber >= len(classifiersParams) and hypEstimation > 0:
                        print("Without grid/random search")
                    else:
                        if df_mobile.shape[1] != 0:
                            errT = "no"
                            try:
                                validation = True
                                model = classificationModel(
                                    validation, modelNumber, X_train, y_train, hypEstimation)
                                if validation == True:
                                    print("start-Predict")
                                    X_test_predict = model.predict(X_test)
                                    print("end-Predict")
                                    df_stats_model = printFinalTable(df_stats_model, initialProduct, initialFeature, outlier, selectFeature, model,
                                                                     modelNumber, X_test_predict, y_test, df_mobile, hypEstimation)
                            except ValueError:
                                print("This is an error message!")
                                errT = "jj"
                            print("OT ", str(outlier) + " z " +
                                  str(len(outliersName)) + " SF ", str(selectFeature) + "z" +
                                  str(len(featureSelectionName)) + " MO ", str(modelNumber) +
                                  "z" + str(len(classifiers)) + " HE ", hypEstimation, "error ", errT, " validation ", validation)

                            elapsed_time = time.time() - start_time
                            t = time.localtime()
                            current_time = time.strftime("%H:%M:%S", t)
                            print("end -", current_time)
                            print("all time: ", elapsed_time)
                            print("")


print("last")

df_stats_model.to_csv(r'Information/columns_stats_model.csv',
                      index=False, header=True)


# %%
print(df_stats_model)
cm = sns.light_palette("green", as_cmap=True)

styled = df_stats_model.style.background_gradient(cmap=cm)

styled.to_excel('Information/styled_model.xlsx', engine='openpyxl')

# %%
