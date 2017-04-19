import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


### 1. Read Data ###

def read_data (filename, filetype):
    '''
    Read data from file to pandas DataFrame
        filename, filetype: string
    '''
    if filetype == 'csv':
        return pd.read_csv(filename, index_col=0)
    if filetype == 'xls':
        return pd.read_excel(filename)
    if filetype == 'json':
        return pd.read_json(filename)



### 2. Explore Data ###

def data_overview(df):
    print("___ Summary Statistics ___")
    print(df.describe())
    print("___ Null Values ___")
    print(df.isnull().sum())

def make_graph(df, col_name):
    '''
    Generate a simple graph for the desired column variable
        df: pandas DataFrame
        col_name: desired variable
    '''
    print('Plotting ' + col_name)
    df[col_name].hist()
    plt.title(col_name)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    filename = str(col_name) + '_graph'
    plt.savefig(filename)
    plt.close()
    #plt.show()



### 3. Pre-Process Data ###

def fill_null(df, col_name, fill_method):
    '''
    Fill null values of the specified columnn in the dataframe
        col_name: string
        fill_method: 'mean', 'median'
    '''
    if fill_method == 'mean':
        df[col_name].fillna(value=df[col_name].mean(), inplace=True)  
    if fill_method == 'median':
        df[col_name].fillna(value=df[col_name].median(), inplace=True)
    return df



### 4. Generate Features/ Predictors ###

def discretize_cont_var(df, col_name, num_bins, cut_type, labels):
    '''
    Discretize a continuous variable of the DataFrame
        df: pandas DataFrame
        col_name, cut_type: string
        nnum_bins: integer
        labels: list of strings
    '''
    if cut_type == 'quantile':
        df[col_name +'_discretize'] = pd.qcut(df[col_name], num_bins, labels=labels)
    if cut_type == 'uniform':
        df[col_name +'_discretize'] = pd.cut(df[col_name], num_bins, labels=labels)
    return df       


def binarize_categ_var(df, col_name):
    '''
    Take a categorical variable and create binary/dummy variables from it
        df: pandas DataFrame
        col_name: string, categorical variable to binarize
    '''
    dummies = pd.get_dummies(df[col_name])
    df = df.join(dummies)
    return df



### 5. Build Classifier ###

def split_data(df, X, y, test_size):
    '''
    Split data into training and test sets
        df: Pandas DataFrame
        X: array of string, features
        y: string, outcome variable
        test_size: proportion of the total data used as test dataset

    '''
    X_train, X_test, y_train, y_test  = train_test_split(df[X], df[y], test_size=test_size)
    return X_train, X_test, y_train, y_test    



def test_model(X_train, y_train, features, method):
    '''
    Build classifiers chosen by the user
        X_train, y_train: Pandas DataFrame
        features: list of strings, variables we care about
        method: LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier(),\
        RandomForestClassifier(), GradientBoostingClassifier()
    '''
    X = X_train[features]
    y = y_train
    return method.fit(X, y)



def predict_model(X_train, y_train, X_test, y, features, method):
    '''
    Predict outcomes for test data based on the chosen classifier, and write to csv file
        X_test: test pandas DataFrame
        y: string, outcome variable name, 'SeriousDlqin2yrs'
        features: list of strings, variables we care about
        method: LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier(),\
        RandomForestClassifier(), GradientBoostingClassifier()
    '''
    method.fit(X_train[features], y_train)
    X = X_test[features]
    y_pred = method.predict(X)
    df_pred = X_test
    df_pred[y] = y_pred
    df_pred.to_csv('predictions'+ str(method) + '.csv', header=True)
    


### 6. Evaluate Classifier ###

def eval_model(X_train, y_train, X_test, y_test, features, method):
    method.fit(X_train[features], y_train)
    X = X_test[features]
    y_pred = method.predict(X)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    print('Evaluation for ' + str(method))
    print('Accuracy score is: {}'.format(accuracy))
    print('Recall score is: {}'.format(recall))
    print('Precision score is: {}'.format(precision))
    print('________#############_______')


