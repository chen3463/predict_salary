
import os
import gc
import math
import numpy as np 
import pandas as pd 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.preprocessing import LabelEncoder
from hyperopt import hp
from hyperopt import STATUS_OK
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

class LinearModel_Engine:
    """
    A light GBM engine
    """

    def __init__(self, 
                 LM,
                 X_train, 
                 Y_train, 
                 X_test = None, 
                 Y_test = None, 
                 encode = False, 
                 MAX_EVALS = 50):
        """
        Initialize 
        """
        if LM == 'lasso':
            self.LM = Lasso 
        if LM == 'ridge':
            self.LM = Ridge
        if LM == 'EN':
            self.LM = ElasticNet    
        self.X_train = X_train 
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.encode = encode # True if one hot encoding else label encoding and LightGBM will handle categorical varaibles   
        self.MAX_EVALS = MAX_EVALS 
        
    def check_RMSE(self, y, pred):
        RMSE = math.sqrt(mean_squared_error(y, pred))
        return RMSE

    def get_categorical_variables(self):
        '''
        return a list of categorical variables in training data sets 
        '''
        var_cate = [] # to save names of categorical variables
        for col in self.X_train.columns:
            if (self.X_train[col].dtype == 'object') and (self.X_train[col].nunique() < self.X_train.shape[0]):
                var_cate.append(col)
        return var_cate

    def get_datasets(self):
        
        var_cate = self.get_categorical_variables()
        
        if self.X_test is not None and self.Y_test is not None:
            features_DEV, features_OOT, labels_DEV, labels_OOT = self.X_train, self.X_test, self.Y_train, self.Y_test
        else:
            features_DEV, features_OOT, labels_DEV, labels_OOT = train_test_split(self.X_train, self.Y_train, test_size = 0.3, \
                                                      stratify=self.Y_train, random_state=100)
        
        
        gc.collect()
            
        return var_cate, features_DEV, features_OOT, labels_DEV, labels_OOT
            
    def encoding(self, var_cate, features_DEV, features_OOT):
        # print(var_cate)
        if var_cate is not None:
            if self.encode: # one hot encoding
                features = pd.get_dummies(features_DEV, prefix = var_cate, columns = var_cate)
                features1 = pd.get_dummies(features_OOT, prefix = var_cate, columns = var_cate)
                # Align the dataframes by the columns by the columns
                DEV_set, OOT_set = features.align(features1, join = 'left', axis = 1)


            else: # start label encoding
                for col in var_cate:
                    le = LabelEncoder()
                    le.fit(features_DEV[col])
                    features_DEV[col] = le.transform(features_DEV[col])
                DEV_set = features_DEV

                for col in var_cate:
                    le = LabelEncoder()
                    le.fit(features_OOT[col])
                    features_OOT[col] = le.transform(features_OOT[col])
                OOT_set = features_OOT
                
        self.feature_names = features_DEV.columns # keep varibles names 
        
        return DEV_set, OOT_set, features_DEV.columns
        
    
    def train(self, X_train, y_train, X_valid, y_valid, hyperparameters):

        start = timer()
        if self.LM is not ElasticNet:
            model = self.LM(normalize = True, alpha = hyperparameters['alpha'], random_state = 100).fit(X_train, y_train)
        elif self.LM is ElasticNet:
            model = self.LM(normalize = True, alpha = hyperparameters['alpha'], l1_ratio = hyperparameters['l1_ratio'], random_state = 100).fit(X_train, y_train)
        else:
            raise Exception('wrong model')
        pred_train = model.predict(X_train)
        pred_val = model.predict(X_valid)
        DEV_metric = self.check_RMSE(y_train, pred_train)
        OOT_metric = self.check_RMSE(y_valid, pred_val)

        run_time = timer() - start
        
        return model, hyperparameters, DEV_metric, OOT_metric, run_time
        
        
    def bayesian_objective(self, hyperparameters):
        '''
        Objective function for bayesian optimization search for hyperparameters tuning
        TTD: DEV + VAL samples
        '''

        var_cate, features_DEV, features_OOT, y_train, y_valid = self.get_datasets()
        
        X_train, X_valid, _ = self.encoding(var_cate, features_DEV, features_OOT)
        

        model, hyperparameters, DEV_metric, OOT_metric, run_time = self.train(X_train, y_train, X_valid, y_valid, hyperparameters) 

        loss = OOT_metric   

        return {'loss': loss, 'hyperparameters': hyperparameters,  'iteration': ITERATION, 
                'training_time': run_time, 'status': STATUS_OK, 
                'attachments': {'DEV_metric': DEV_metric, 
                                'OOT_metric': OOT_metric, 
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
            model_dir = 'best_model.txt'

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

        best_ind = metrics_records.loc[:,'OOT_metric'].idxmin()
      

        pickle.dump(trials.trial_attachments(trials.trials[best_ind])['model'], open(model_dir, 'wb'))

        return metrics_records.iloc[best_ind, :], \
                trials.trials[best_ind]['result']['hyperparameters'], \
                trials.results[best_ind]