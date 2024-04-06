###########################################################################
#                       Functions for complete EDA                        #
#-------------------------------------------------------------------------#
#   - get_column_type for discrimine variables in numerical or categorical#
#   - explore for get basic info and both list of numerical and           #
#     categorical variables                                               #
#   - FindDuplicates for find and drop (or not) duplicates based in one   #
#     given variable                                                      #
#   - univariate_hist for plot histograms of a list of variables          #
#   - univariate_histbox for plot histograms and boxplotof a list of      #
#     variables                                                           #
#   - multivariate_barplots for multivariate barplots by count or mean    #
#   - factorize_categorical for factorize list of categorical variables   #
#   - correlation_matrix for build correlation matix in heatmap           #
#   - numerical_box for plot boxplot of numerical variables from a list   #
#   - outliers_iqr for drop or sustitute outliers in a iqr*sigma          #
#   - splitter for split in train and test all csv from a root and save   #
#   - normalize for normalice X_train and X_test by any scaler of         #
#     sklearn.preprocessing module                                        #
#   - feature_sel for make feature selection by any method and with any   #
#     test included in sklearn.feature_selection                          #
###########################################################################
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
from sklearn.model_selection import train_test_split
import importlib
import os

def get_column_type(series):
    if pd.api.types.is_numeric_dtype(series):
        return 'Numeric'
    else:
        return 'Categorical'

def explore(data_frame):    
    # Get shape
    num_rows, num_columns = data_frame.shape
    print('Rows:', num_rows)
    print('Columns:', num_columns)

    # Get null and type info
    column_data = pd.DataFrame({
        'Non-Null Count': data_frame.count(),
        'Null Count': data_frame.isnull().sum(),
        'Data Type': data_frame.dtypes
    }) 
    # Add if a variable is categorical or numerical  
    column_data['Data Category'] = data_frame.apply(get_column_type)
    print(tabulate(column_data, headers='keys', tablefmt='pretty'))

    # Get list of categorical and numerical variables
    categorical_columns = list(column_data[column_data['Data Category'] == 'Categorical'].index)
    numeric_columns = list(column_data[column_data['Data Category'] == 'Numeric'].index)
    
    return categorical_columns, numeric_columns

def FindDuplicates(data_frame, id_col, Drop=False):    
    if Drop == 'True':
        deduplicated_df = data_frame.drop_duplicates(data_frame.columns.difference([id_col]))
        if len(deduplicated_df) < len(data_frame):
            print(f" {len(data_frame) - len(deduplicated_df)} duplicates have been removed")
        return deduplicated_df
    else:
        duplicates_count = data_frame.duplicated(subset=[id_col]).sum()
        if duplicates_count > 0:
            print(f" {duplicates_count} duplicates have been found")
        return duplicates_count

def univariate_hist(variables, data_frame, color_='#1295a6', kde_=False):
    num_plots = len(variables)
    num_rows = (num_plots - 1) // 3 + 1  # N rows
    
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5*num_rows))
    axes = axes.flatten()  

    for i, var in enumerate(variables):
        ax = axes[i]
        sns.histplot(data_frame[var], ax=ax, kde=kde_, color=color_)
        ax.set_title(var)
        ax.set_xlabel('')
        ax.set_ylabel('')
        
    # Remove empty axes
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()

def univariate_histbox(variables, data_frame, color_='#1295a6'):
    num_variables = len(variables)
    num_rows = math.ceil(num_variables / 3)
    
    fig, axis = plt.subplots(2 * num_rows, 3, figsize=(18, 6 * num_rows), gridspec_kw={'height_ratios': [8, 4] * num_rows})

    for i, col in enumerate(variables):
        row = i // 3
        col_in_row = i % 3
        
        sns.histplot(data_frame, x=col, ax=axis[row * 2, col_in_row], color=color_)
        sns.boxplot(data_frame, x=col, ax=axis[row * 2 + 1, col_in_row], color=color_)
    
    if num_variables % 3 != 0:
        for j in range(num_variables % 3, 3):
            axis[num_rows * 2 - 1, j].remove()
            axis[num_rows * 2 - 2, j].remove()

    plt.tight_layout()

def multivariate_barplots(df, variable_lists,y_='count',palette_='Set2'):
    # df=dataframe
    # variable_lists: list of list like [x variable, y variable, hue variable]
    # y_ : y values calculated as 'count' or 'mean'

    num_plots = len(variable_lists)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots))

    for i, variables in enumerate(variable_lists):
        if y_ == 'count':
            df_ = df.groupby(variables).size().reset_index(name='count')

            x, hue = variables[0], variables[2]
            ax = axes[i]
            sns.barplot(data=df_, x=x, y='count', hue=hue, ax=ax,palette=palette_,errorbar=None)
            ax.set_xlabel(variables[0])
            ax.set_ylabel('Count')
        elif y_ == 'mean':
            try:
                mean_ = df.groupby([variables[0], variables[2]])[variables[1]].mean().reset_index()
                x, hue = variables[0], variables[2]
                ax = axes[i]
                sns.barplot(data=mean_, x=x, y=variables[1], hue=hue, ax=ax,palette=palette_,errorbar=None)
                ax.set_xlabel(variables[0])
                ax.set_ylabel('Mean')
            except:
                print('y variable is not numerical')
    plt.tight_layout()

def factorize_categorical(df,variables_list):
    for var in variables_list:
        df[var] = pd.factorize(df[var])[0]
    return df

def correlation_matrix(df, variables_list):
    fz_df = factorize_categorical(df,variables_list)

    fig, axis = plt.subplots(figsize = (15, 15))
    sns.heatmap(fz_df.corr(), annot = True, fmt = ".2f")

def numerical_box(variables, data_frame, color_='#1295a6'):
    num_plots = len(variables)
    num_rows = (num_plots - 1) // 3 + 1  # N rows
    
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5*num_rows))
    axes = axes.flatten()  

    for i, var in enumerate(variables):
        ax = axes[i]
        sns.boxplot(x=data_frame[var], ax=ax, color=color_)
        ax.set_title(var)
        ax.set_xlabel('')
        ax.set_ylabel('')
        

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()

def outliers_iqr(df,var,sigma,Do='nothing'):
    #df: dataframe
    #var: variable
    #sigma: tolerance for iqr
    #Do: 'drop' for eliminate outliers rows
        # 'mode' for replace by mode
        # 'mean' for replace by mean
        # 'median' for replace by median
    df_ = df.copy()
    descr = df_[var].describe()
 
    iqr = descr["75%"] - descr["25%"]
    upper_l = descr["75%"] + sigma*iqr
    lower_l = descr["25%"] - sigma*iqr
    print('Upper_l:', upper_l, 'Lower_l:', lower_l)

    outliers = df_[(df_[var] >= upper_l) | (df_[var] < lower_l)]
    num_outliers = outliers.shape[0]     

    if Do == 'nothing':
        print(str(num_outliers)+' outliers have been found')
        pass
    else:
        if Do != 'drop':
            if Do == 'mode':
                replacer = df_[var].mode()                    
            elif Do == 'mean':
                replacer = df_[var].mean()
            elif Do == 'median':
                replacer = df_[var].median()

            replace_func = lambda x: x if lower_l <= x < upper_l else replacer
            df_[var] = df_[var].apply(replace_func)        
            print(str(num_outliers)+' outliers have been treated by replacing them with the '+Do)
        else:
            df_ = df_[df_[var].between(lower_l, upper_l)]
            print(str(num_outliers)+' outliers have been treated by dropping')
    return outliers,df_
    

def splitter(origin_root,predictors,target):    
    csv_files = []

    for file in os.listdir(origin_root):
        if file.endswith(".csv"):
            csv_files.append(os.path.join(origin_root, file))
    
    for df_root in csv_files:
        df = pd.read_csv(df_root)
        X = df[predictors]
        Y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)
        
        name = (df_root.split('/'))[-1].split('.')[0]       
        destino =origin_root+'SplitData/'
        if not os.path.exists(destino):
            os.makedirs(destino)

        X_train.to_csv(destino+name+'_Xtrain.csv', index=False)
        X_test.to_csv(destino+name+'_Xtest.csv', index=False)
        y_train.to_csv(destino+name+'_ytrain.csv', index=False)
        y_test.to_csv(destino+name+'_ytest.csv', index=False)
                      
def normalize(origin_root,predictors,scaler_='StandardScaler'):
    # origin_root: root of X_train, X_test in processed/SplitData
    # predictors: variables except target (numeric)
    # scaler: any scaler from sklearn.preprocessing
    
    module = importlib.import_module('sklearn.preprocessing')
    submodule = getattr(module, scaler_)
    scaler = submodule()
    csv_files = []

    for file in os.listdir(origin_root):
        if file.endswith(".csv"):
            csv_files.append(os.path.join(origin_root, file))
    
    for df_root in csv_files:
        if '_X' in str(df_root):
            df = pd.read_csv(df_root)

            scaler.fit(df)
            scaler_df = scaler.transform(df)
            scaler_df = pd.DataFrame(scaler_df,index = df.index, columns = predictors)
            
            name = (df_root.split('/'))[-1].split('.')[0]       
            destino =origin_root+'NormData/'
            if not os.path.exists(destino):
                os.makedirs(destino)

            scaler_df.to_csv(destino+name+'_norm.csv', index=False)

def feature_sel(X_train_,y_train_,k_,file_name,method_='SelectKBest', test_='mutual_info_classif'):
    module = importlib.import_module('sklearn.feature_selection')
    method = getattr(module, method_)
    test = getattr(module, test_)
    test_to_apply = method(test)
        
    selection_model = method(test, k = k_)
    selection_model.fit(X_train_, y_train_)
    ix = selection_model.get_support()
    X_train_sel = pd.DataFrame(selection_model.transform(X_train_), columns = X_train_.columns.values[ix])

    if not os.path.exists('../data/processed/SplitData/FeatureSel/'):
            os.makedirs('../data/processed/SplitData/FeatureSel/')
    X_train_sel.to_csv('../data/processed/SplitData/FeatureSel/'+str(file_name)+'_FeatureSel.csv', index=False)

    return X_train_sel