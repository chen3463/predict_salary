import os
import gc

import numpy as np 
import pandas as pd 
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.preprocessing import LabelEncoder
from hyperopt import hp
from hyperopt import STATUS_OK
import matplotlib.pyplot as plt

class LightGBM_Engine:
    """
    A light GBM engine
    """

    def __init__(self, X_train, 
                 Y_train, 
                 X_test = None, 
                 Y_test = None, 
                 encode = False, 
                 weight = None, 
                 metric = None,
                 Eval = None, 
                 P_keep = None, 
                 N_top = 0, 
                 MAX_EVALS = 50,
                 n_jobs = 2):
        """
        Initialize the light GBM 
        """
        self.X_train = X_train 
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.Eval = Eval
        self.metric = metric
        self.encode = encode # True if one hot encoding else label encoding and LightGBM will handle categorical varaibles
        self.weight = weight # Not None if reject inference is applied
        self.Eval = Eval
        

        self.P_keep = P_keep
        self.N_top = N_top
            
        self.MAX_EVALS = MAX_EVALS 
        self.bins = 10 # for KS in metric
        self.n_jobs = n_jobs
        
    '''
    Calculate and rank feature importance
    '''
    def rank_feature_importances(self, df):
        """
        Plot importances returned by a model. This can work with any measure of
        feature importance provided that higher importance is better. 

        Args:
            df (dataframe): feature importances. Must have the features in a column
            called `features` and the importances in a column called `importance

        Returns:

            df (dataframe): feature importances sorted by importance (highest to lowest) 
            with a column for normalized importance
            """

        # Sort features according to importance
        df = df.sort_values('importance', ascending = False).reset_index()

        # Normalize the feature importances to add up to one
        df['importance_normalized'] = df['importance'] / df['importance'].sum()

        return df
    
    def get_categorical_variables(self):
        '''
        return a list of categorical variables in training data sets 
        '''
        var_cate = [] # to save names of categorical variables
        for col in self.X_train.columns:
            if (self.X_train[col].dtype == 'object') and (self.X_train[col].nunique() < self.X_train.shape[0]):
                var_cate.append(col)
                # self.X_train[col] = self.X_train[col].astype('category')
                # self.X_train[col], _ = pd.factorize(self.X_train[col])
        return var_cate

    def get_datasets(self):
        
        var_cate = self.get_categorical_variables()
        
        if self.X_test is not None and self.Y_test is not None:
            features_DEV, features_OOT, labels_DEV, labels_OOT = self.X_train, self.X_test, self.Y_train, self.Y_test
        else:
            features_DEV, features_OOT, labels_DEV, labels_OOT = train_test_split(self.X_train, self.Y_train, test_size = 0.3, \
                                                      stratify=self.Y_train, random_state=100)
        
        
        gc.collect()
        ### if data generated srom reject sampling, every sample has a weight ###
        if self.weight:
            weights_DEV = features_DEV[self.weight]
            weights_OOT = features_OOT[self.weight]
            features_DEV = features_DEV.drop(self.weight, axis=1)
            features_OOT = features_OOT.drop(self.weight, axis=1)
        
        else:
            weights_DEV = None
            weights_OOT = None
            
        return var_cate, features_DEV, features_OOT, labels_DEV, labels_OOT, weights_DEV, weights_OOT
            
    def encoding(self, var_cate, features_DEV, features_OOT, labels_DEV, labels_OOT, weights_DEV, weights_OOT):
        # print(var_cate)
        if var_cate is not None:
            if self.encode: # one hot encoding
                features = pd.get_dummies(features_DEV, prefix = var_cate, columns = var_cate)
                features1 = pd.get_dummies(features_OOT, prefix = var_cate, columns = var_cate)
                # Align the dataframes by the columns by the columns
                features_DEV, features_OOT = features.align(features1, join = 'left', axis = 1)
                self.DEV_set = lgb.Dataset(data=features_DEV, label = labels_DEV, weight=weights_DEV,
                                          free_raw_data=False)
                self.OOT_set = lgb.Dataset(data=features_OOT, label = labels_OOT, weight=weights_OOT,
                                          free_raw_data=False)

            else: # start label encoding
                for col in var_cate:
                    le = LabelEncoder()
                    le.fit(features_DEV[col])
                    features_DEV[col] = le.transform(features_DEV[col])

                for col in var_cate:
                    le = LabelEncoder()
                    le.fit(features_OOT[col])
                    features_OOT[col] = le.transform(features_OOT[col])

                self.DEV_set = lgb.Dataset(data=features_DEV, label = labels_DEV, 
                                           weight=weights_DEV, categorical_feature = var_cate,
                                           free_raw_data=False)
                self.OOT_set = lgb.Dataset(data=features_OOT, label = labels_OOT, 
                                           weight=weights_OOT, categorical_feature = var_cate,
                                           free_raw_data=False)
                
        self.feature_names = features_DEV.columns # keep varibles names 
        
        return self.DEV_set, self.OOT_set, features_DEV.columns
        
    
    def train(self, DEV_set, OOT_set, hyperparameters):
        
        hyperparameters['random_state'] = 100
        hyperparameters['n_jobs'] = self.n_jobs
        hyperparameters['verbosity'] = -1
        hyperparameters['n_estimators'] = 2000

        evals_results = {}

        start = timer()
        if self.Eval == None:

            hyperparameters['metrics'] = self.metric
            model = lgb.train(hyperparameters, DEV_set, early_stopping_rounds = 50, \
                              evals_result = evals_results,\
                              verbose_eval = False, \
                              valid_sets = [OOT_set, DEV_set], \
                              valid_names = ['OOT', 'DEV'])

            DEV_metric = evals_results['DEV'][self.metric][model.best_iteration - 1]
            OOT_metric = evals_results['OOT'][self.metric][model.best_iteration - 1]

        else:

            model = lgb.train(hyperparameters, DEV_set, early_stopping_rounds = 50, \
                              evals_result = evals_results, feval = self.Eval, \
                              verbose_eval = False, \
                              valid_sets = [OOT_set, DEV_set], \
                              valid_names = ['OOT', 'DEV'])

            DEV_metric = evals_results['DEV']['KS_Eval'][model.best_iteration - 1]
            OOT_metric = evals_results['OOT']['KS_Eval'][model.best_iteration - 1]

        run_time = timer() - start

        hyperparameters['n_estimators'] = model.num_trees() 
        if 'metrics' in hyperparameters.keys():
            del hyperparameters['metrics']

        if 'n_jobs' in hyperparameters.keys():
            del hyperparameters['n_jobs']

        if 'verbosity' in hyperparameters.keys():
            del hyperparameters['verbosity']

        if 'random_state' in hyperparameters.keys():
            del hyperparameters['random_state']  
        
        return model, evals_results, hyperparameters, DEV_metric, OOT_metric, run_time
        
        
    def bayesian_objective(self, hyperparameters):
        '''
        Objective function for bayesian optimization search for hyperparameters tuning
        TTD: DEV + VAL samples
        '''

        var_cate, features_DEV, features_OOT, labels_DEV, labels_OOT, weights_DEV, weights_OOT = self.get_datasets()
        
        DEV_set, OOT_set, feature_names = self.encoding(var_cate, features_DEV, features_OOT, labels_DEV, labels_OOT, weights_DEV, weights_OOT)
        

        model, _, hyperparameters, DEV_metric, OOT_metric, run_time = self.train(DEV_set, OOT_set, hyperparameters) 

        
        feature_importance_values = model.feature_importance()
        feature_importance = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

        feature_importance = self.rank_feature_importances(feature_importance)

        if self.P_keep: # keep top P percent variables

            stop = 0

            for i in range(len(feature_names)):
                if sum(feature_importance.importance_normalized[0:i]) >= self.P_keep:
                    stop = i
                    break

            print(sum(feature_importance.importance_normalized[0:i])*100, '% contribution kept')
            var = list(feature_importance.loc[0:stop, 'feature'])

        elif self.N_top:
            var = list(feature_importance.loc[0:self.N_top, 'feature']) # keep top N variables
        else:
            var = list(feature_importance.loc[0:, 'feature'])

        # if on_hot encoding is being used, remove the whole varible even only 1 category was in deletd varibles  
        if self.encode:
            var_cat_save = []
            var_del = []
            for col in var_cate:
                for item in var:
                    if item.find(col) >= 0:
                        var_cat_save = var_cat_save + [col]
                        var_del = var_del + [item]

            var = [x for x in var if x not in list(set(var_del))]
            var = var + list(set(var_cat_save))

        if self.metric in ('rmse', 'mse'):
            loss = OOT_metric   
        else:
            loss = -OOT_metric
        
        
        return {'loss': loss, 'hyperparameters': hyperparameters,  'iteration': ITERATION, 
                'training_time': run_time, 'status': STATUS_OK, 
                'attachments': {'DEV_metric': DEV_metric, 
                                'OOT_metric': OOT_metric, 
                                'selected_vars': var,
                                'feature': feature_importance.feature, 
                                'feature_importance': feature_importance.importance_normalized,
                                'model': model}
               }   
    
    
    def evaluation(self, space, out_file=None, model_dir=None):
        '''
        evaluation function for bayesian search
        '''
        import random
        from hyperopt import tpe
        from hyperopt import Trials
        from hyperopt import fmin
        random.seed(4)
        if model_dir is None:
            model_dir = 'best_mdole.txt'

        trials = Trials()

        global ITERATION
        ITERATION = 0

        best = fmin(fn = self.bayesian_objective, space = space, algo = tpe.suggest, trials = trials,
                max_evals = self.MAX_EVALS, rstate = np.random.RandomState(100))

        DEV_metric = []
        OOT_metric = []

        for i in range(self.MAX_EVALS):
            DEV_metric.append(trials.trial_attachments(trials.trials[i])['DEV_metric'])
            OOT_metric.append(trials.trial_attachments(trials.trials[i])['OOT_metric'])

        metrics_records = pd.DataFrame({'Itr': list(range(0, self.MAX_EVALS)), 
                       'DEV_metric': DEV_metric, 
                       'OOT_metric': OOT_metric
                       })

        keys = trials.trials[0]['result']['hyperparameters'].keys()
        hyperparameters_records= pd.DataFrame(index=range(self.MAX_EVALS), columns=keys)

        for i in range(self.MAX_EVALS):
            hyperparameters_records.iloc[i,:] = list(trials.trials[i]['result']['hyperparameters'].values())

        all_records = pd.concat([metrics_records, hyperparameters_records], axis=1)

        if out_file:
            all_records.to_csv(out_file, index=False)

        if self.metric in ('rmse', 'msr'):
            best_ind = metrics_records.loc[:,'OOT_metric'].idxmin()
        else:
            best_ind = metrics_records.loc[:,'OOT_metric'].idxmax()

        trials.trial_attachments(trials.trials[best_ind])['model'].save_model(model_dir)

        return metrics_records.iloc[best_ind, :], \
                trials.trial_attachments(trials.trials[best_ind])['feature'], \
                trials.trial_attachments(trials.trials[best_ind])['feature_importance'], \
                trials.trial_attachments(trials.trials[best_ind])['selected_vars'], \
                trials.trials[best_ind]['result']['hyperparameters'], \
                trials.results[best_ind]