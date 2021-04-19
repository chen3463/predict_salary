import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.metrics import mean_squared_error

'''
check percentage of missing values and return columns with NA and percentages
'''
def miss_val_per(df, imput=True):
    pct = df.isnull().mean() * 100
    cols = [key for key, value in pct.items() if value > 0]
    print(cols)
    for col in cols:
        col_type = df[col].dtypes
        print(col + ": " + col_type.name)  
        
def check_var_type(df):
    # Number of each type of column
    print(df.dtypes.value_counts())
    # Number of unique classes in each object column
    print(df.select_dtypes('object').apply(pd.Series.nunique, axis = 0))


def get_categorical_variables(X_train):
    '''
    return a list of categorical variables in training data sets 
    '''
    var_cate = [] # to save names of categorical variables
    for col in X_train.columns:
        if (X_train[col].dtype == 'object') and (X_train[col].nunique() < X_train.shape[0]):
            var_cate.append(col)
            # X_train[col] = X_train[col].astype('category')
            # X_train[col], _ = pd.factorize(X_train[col])
    return var_cate

def describe_group(df, var, target):

    df1 = pd.DataFrame(group.describe().rename(columns={target: i}).squeeze()
            for i, group in df[[var, target]].groupby(var))

    return df1

def plot_categorical_vars(df, categorical_vars):

    #  Categorical Data
    a = 3  # number of rows
    b = 2  # number of columns
    c = 1  # initialize plot counter

    plt.figure(figsize=(a * 5, b * 8))
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=.5, wspace=0.4)
    for col in categorical_vars:
        plt.subplot(a, b, c)
        plt.title("Distribution of categorical varible {}".format(col))
        df[col].value_counts().plot.bar()
        plt.xticks(rotation = 90)
        c = c + 1

    plt.show()

def plot_categorical_target(df, categorical_vars, target):

    a = 5  # number of rows
    b = 2  # number of columns
    c = 1  # initialize plot counter
    plt.figure(figsize=(a * 5, b * 20))
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=.5, wspace=0.4)

    for col in categorical_vars:
        
        plt.subplot(a, b, c)
        plt.title('Distribution of {}'.format(col))
        df[col].value_counts().plot.bar()
        c = c + 1

        plt.subplot(a, b, c)
        plt.title("{}".format(col) + " vs. " + "{}".format(target))
        plt.xlabel(col)
        sns.boxplot(x = col, y = target, data = df.sort_values(target))
        plt.xticks(rotation = 90)
        c = c + 1

    plt.show()



def plot_numerical_target(df, numerical_vars, target):   

    a = 3  # number of rows
    b = 2  # number of columns
    c = 1  # initialize plot counter

    plt.figure(figsize=(a * 5, b * 8))
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace = 1, wspace = 0.4)
    for col in numerical_vars:
        plt.subplot(a, b, c)
        plt.title("{}".format(target) + " vs. " + "{}".format(col))
        plt.xlabel(col)
        plt.ylabel(target)
        plt.scatter(x = col, y= target, data = df, s = 1)
        plt.xticks(rotation = 90)
        c = c + 1
    plt.show()
    

def drop_vars(vars_drop, X_train=None, X_val=None, X_test=None):
    if X_train is not None:
        X_train = X_train.drop(vars_drop, axis = 1)
    if X_val is not None:
        X_val = X_val.drop(vars_drop, axis = 1)
    if X_test is not None:
        X_test = X_test.drop(vars_drop, axis = 1)
    return X_train, X_val, X_test

def encoding(encode, var_cate, features_DEV, features_OOT):

    if var_cate is not None:
        if encode == 'one hot': # one hot encoding
            features_DEV = pd.get_dummies(features_DEV, prefix = var_cate, columns = var_cate)
            features_OOT = pd.get_dummies(features_OOT, prefix = var_cate, columns = var_cate)
            # Align the dataframes by the columns by the columns
            features_DEV, features_OOT = features_DEV.align(features_OOT, join = 'left', axis = 1)

        else: # start label encoding
            for col in var_cate:
                le = LabelEncoder()
                le.fit(features_DEV[col])
                features_DEV[col] = le.transform(features_DEV[col])

            for col in var_cate:
                le = LabelEncoder()
                le.fit(features_OOT[col])
                features_OOT[col] = le.transform(features_OOT[col])
    
    return features_DEV, features_OOT

def hist_kdp(x):
    # Density Plot and Histogram of all arrival delays
    sns.distplot(x, hist=True, kde=True, 
                 bins=int(180/5), color = 'darkblue', 
                 hist_kws={'edgecolor':'black'},
                 kde_kws={'linewidth': 4})

def check_target(df, target):
    plt.figure(figsize = (15, 6))
    plt.subplot(1,2,1)
    sns.boxplot(df[target])
    plt.subplot(1,2,2)
    # sns.distplot(training['salary'], bins=20,color = 'b')
    hist_kdp(df[target])
    plt.show()

def VIF(X_train):    
    # VIF dataframe 
    vif_data = pd.DataFrame() 
    vif_data["feature"] = X_train.columns
    
    # calculating VIF for each feature 
    vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) 
                            for i in range(len(X_train.columns))] 
    
    print(vif_data)

def check_RMSE(model, y, pred):
    print("RMSE of model {} is: ".format(model), math.sqrt(mean_squared_error(y, pred)))

def plot_importance(df, top=15):    
    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:top]))), 
            df['importance_normalized'].head(top), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:top]))))
    ax.set_yticklabels(df['feature'].head(top))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()

