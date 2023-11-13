import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold , GroupKFold
import joblib
import pandas as pd 
def recursive_feature_elimination(session,model_id,model,x_train,y_train,x_test,task):
    feature_engineering_tracking_dict=joblib.load(f"{session}/feature_engineering_tracking.joblib")
    feature_engineering_tracking_dict["feature_selection"]="recursive_feature_elimination with cross validation (RFECV)"
    selector=None
    if task=="regression":
        groups = [i // 10 for i in range(x_train.shape[0])] 
        n_splits = len(set(groups))
        # create a group-wise cross-validation object
        cv = GroupKFold(n_splits=n_splits)
        selector = RFECV(estimator=model, step=1, cv=cv, scoring='neg_mean_squared_error')
        # Fit the RFE object to the data
        selector.fit(x_train,y_train,groups=groups)
    else:
        number_of_splits=int(y_train[y_train.columns].nunique()[y_train.columns])
        cv=StratifiedKFold(n_splits=number_of_splits)
        selector = RFECV(estimator=model, step=1, cv=cv, scoring='accuracy') 
        # Fit the RFE object to the data
        selector.fit(x_train,y_train)
    feature_engineering_tracking_dict["selected_features"]=int(selector.n_features_)
    feature_names = x_train.columns[selector.get_support()]  
    x_train_selected = selector.transform(x_train)
    x_test_selected = selector.transform(x_test) 
    x_train_selected_df = pd.DataFrame(x_train_selected, columns=feature_names)
    x_test_selected_df = pd.DataFrame(x_test_selected, columns=feature_names)
    feature_engineering_tracking_dict["selected_features_names"]=x_train_selected_df.columns.tolist()
    joblib.dump(feature_engineering_tracking_dict,f"{session}/{model_id}/feature_engineering_tracking.joblib")   
    joblib.dump(selector,f'{session}/{model_id}/recursive_feature_elimination.joblib')
    return x_train_selected_df,x_test_selected_df
