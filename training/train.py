from sklearn.model_selection import train_test_split
from logistic_regression_model import logistic_regression
from svm_model import svm_model
from linear_regression_model import linear_regression
from feature_engineering import feature_selection
from kmeans_model import kmeans_model
from naive_bayes_model import naive_bayes
from decision_trees_model import decision_trees
from xgboost_regression_model import xgb_regression
from xgboost_classification_model import xgboost_classifier
from randomforest_classification_model import randomforest_classifier
from randomforest_regression_model import randomforest_regression
from decision_tree_regression_model import decisiontree_regression
from mlp_classifier_model import mlp_classifier
import os 
import sys
import pandas as pd
import shutil
import joblib
# Get the current working directory
path=os.getcwd()
# Get the parent directory of the current working directory
dataset_path_prefix = os.path.dirname(path)+"/"
data_tools_path = os.path.join(path,'data_tools')
tools_path = os.path.join(path,'tools')
sys.path.insert(0, data_tools_path)
sys.path.insert(0, tools_path)
from supported_data_formats import read_supported
from db_tools import get_dataset,save_classes,edit_trained_model,save_split_strategy,change_train_session_status
from data_preprocessing import encoder,scaler_clustering,scaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import ExtraTreeClassifier, DecisionTreeRegressor
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from celery import shared_task
from celery.result import AsyncResult

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.neural_network import MLPClassifier
import json
import logging

logger = logging.getLogger("main")

def classification_model_param_check(user_id,session,task,model_id,id_dataset,x_train,y_train,x_test,y_test,param_dict,apply_feature_selection=False):
 
    parent_folder_path=session+"/"
    folder_name=str(model_id)
    path=os.path.join(parent_folder_path, folder_name)
    os.mkdir(path)
    if model_id == 4:
        model= svm_model
        model_name="svm"
        feature_selection_model=SVC(kernel="linear")
    elif model_id == 1:
        model_name="logistic regression"
        if param_dict["param_space"]["penalty"]==[None] :
            param_dict["param_space"]["penalty"]=["none"]
        model= logistic_regression
        feature_selection_model=LogisticRegression()
    elif model_id == 2:
        model_name="naive bayes"
        model= naive_bayes
        feature_selection_model=GaussianNB()
    elif model_id == 3:
        model_name="decision trees"
        model= decision_trees
        feature_selection_model= ExtraTreeClassifier()       
    elif model_id == 6:
        model_name="xgboost classifier"
        model= xgboost_classifier 
        feature_selection_model=XGBClassifier()
    elif model_id == 9:
        model_name="Random Forest Classifier"
        model= randomforest_classifier
        feature_selection_model= RandomForestClassifier() # FIXME: why use model as feature selection
    elif model_id == 11:
        model_name="Multilayer perceptron"
        model= mlp_classifier
        feature_selection_model= MLPClassifier(early_stopping=True)
       
    if x_train.shape[1] >= 2 and apply_feature_selection:    
        print(f"FS started in {model_name}")
        logger.info(f"model {model_name} feature selection is launched ",extra={'user_id': user_id}) 
        x_train,x_test=feature_selection(session,model_id,x_train,task,model_name,y_train,x_test,feature_selection_model)
    else:
        logger.info(f"model {model_name} feature selection is skipped ",extra={'user_id': user_id}) 
        feature_engineering_tracking_dict=joblib.load(f"{session}/feature_engineering_tracking.joblib")
        feature_engineering_tracking_dict["feature_selection"]="Default"
        feature_engineering_tracking_dict["selected_features"]=x_train.shape[1]
        feature_engineering_tracking_dict["selected_features_names"]=x_train.columns.tolist()
        joblib.dump(feature_engineering_tracking_dict,f"{session}/{model_id}/feature_engineering_tracking.joblib")   
    if  len(param_dict["optimizer"])!=0:
        logger.info(f"model {model_name} is launched ",extra={'user_id': user_id})     
        model(user_id,session,model_id,id_dataset,x_train,y_train,x_test,y_test,param_dict["n_iter"],optimizer=param_dict["optimizer"],param_space=param_dict["param_space"])
    else:
        logger.info(f"model {model_name} is launched ",extra={'user_id': user_id})     
        model(user_id,session,model_id,id_dataset,x_train,y_train,x_test,y_test,param_dict["n_iter"],param_space=param_dict["param_space"])
              
def regression_model_param_check(user_id,session,task,model_id,id_dataset,x_train,y_train,x_test,y_test,param_dict,apply_feature_selection=False):
    parent_folder_path=session+"/"
    folder_name=str(model_id)
    path=os.path.join(parent_folder_path, folder_name)
    os.mkdir(path)
    if model_id==5:
        model_name="linear regression"
        model=linear_regression
        feature_selection_model=LinearRegression()
    elif model_id==7:
        model_name="xgb regression"
        model=xgb_regression
        feature_selection_model=XGBRegressor(importance_type="gain")
    elif model_id == 10:
        model_name="Random Forest Regression"
        model= randomforest_regression
        feature_selection_model=RandomForestRegressor()
    elif model_id == 12:
        model_name="Decision Tree Regression"
        model= decisiontree_regression
        feature_selection_model= DecisionTreeRegressor() 
        
       
    if x_train.shape[1] >= 2 and feature_selection_model!=None and apply_feature_selection: 
        logger.info(f"model {model_name} feature selection is launched ",extra={'user_id': user_id}) 
        x_train,x_test=feature_selection(session,model_id,x_train,task,model_name,y_train,x_test,feature_selection_model)
    else:
        logger.info(f"model {model_name} feature selection is skipped ",extra={'user_id': user_id}) 
        feature_engineering_tracking_dict=joblib.load(f"{session}/feature_engineering_tracking.joblib")
        feature_engineering_tracking_dict["feature_selection"]="Default"
        feature_engineering_tracking_dict["selected_features"]=x_train.shape[1]
        feature_engineering_tracking_dict["selected_features_names"]=x_train.columns.tolist()
        joblib.dump(feature_engineering_tracking_dict,f"{session}/{model_id}/feature_engineering_tracking.joblib")   
        
    if len(param_dict["optimizer"])!=0:
        logger.info(f"model {model_name} is launched ",extra={'user_id': user_id}) 
        model(user_id,session,model_id,id_dataset,x_train,y_train,x_test,y_test,param_dict["n_iter"],optimizer=param_dict["optimizer"],param_space=param_dict["param_space"])     
    else:
        logger.info(f"model {model_name} is launched ",extra={'user_id': user_id})     
        model(user_id,session,model_id,id_dataset,x_train,y_train,x_test,y_test,param_dict["n_iter"],param_space=param_dict["param_space"])

def clustering_model_param_check(user_id,session,task,model_id,id_dataset,x,param_dict,apply_feature_selection=False):    
    parent_folder_path=session+"/"
    folder_name=str(model_id)
    path=os.path.join(parent_folder_path, folder_name)
    os.mkdir(path)
    if model_id==8:
        model=kmeans_model
        model_name="KMeans"
    
    if x.shape[1] >= 2: 
        logger.info(f"model {model_name} feature selection is launched ",extra={'user_id': user_id}) 
        x=feature_selection(session,model_id,x,task)
    if len(param_dict["optimizer"])!=0:
        logger.info(f"model {model_name} is launched ",extra={'user_id': user_id})   
        model(user_id,session,model_id,id_dataset,x,param_dict["n_iter"], optimizer=param_dict["optimizer"],param_space=param_dict["param_space"])  
    else:
        logger.info(f"model {model_name} is launched ",extra={'user_id': user_id})   
        model(user_id,session,model_id,id_dataset,x,param_dict["n_iter"],param_space=param_dict["param_space"])
       

@shared_task(bind=True)
def train_models(self,user_id,model_id,id_dataset,param_dict,split_strategy,task,session_id,target=""):

    try:
        file_path, filename = get_dataset(id_dataset)
        file_path=dataset_path_prefix+file_path
        df = read_supported(file_path)
        session=str(session_id)
        if os.path.exists(str(session_id)):
            shutil.rmtree(str(session_id))
        os.mkdir(session)

        print(f"df type: { type(df)}\n df content: {df}")
        task_id=self.request.id
        print(f"task_id content {task_id}\n task_id type {type(task_id)}")
        if task!="clustering":
            if task=="classification":
                x,y,unique_classes= encoder(session,task,df,id_dataset,target)
                unique_classes=json.dumps(unique_classes)
                
                for id in model_id:
                    save_classes(session_id,id_dataset,id,unique_classes)
            else:
                x,y= encoder(session,task,df,id_dataset,target)


            cols= list(x.columns)
            if split_strategy["shuffle"]:
                x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=(split_strategy["train_percentage"]), test_size=split_strategy["test_percentage"], random_state=split_strategy["seed"])
                x_train=pd.DataFrame(data=x_train, columns=cols)
                y_train =pd.DataFrame(data=y_train, columns=[target])
                x_test=pd.DataFrame(data=x_test, columns=cols)
                y_test =pd.DataFrame(data=y_test, columns=[target])
            else:
                x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=(split_strategy["train_percentage"]), test_size=split_strategy["test_percentage"],shuffle=False)
                x_train=pd.DataFrame(data=x_train, columns=cols)
                y_train =pd.DataFrame(data=y_train, columns=[target])
                x_test=pd.DataFrame(data=x_test, columns=cols)
                y_test =pd.DataFrame(data=y_test, columns=[target])
            x_train,x_test=scaler(session,x_train,x_test)
            

        else:
            x=encoder(session,task,df)
            x=scaler_clustering(session,x)
        if task=="classification":

            i=0
            for id in model_id:
                params=param_dict[i]
                classification_model_param_check(user_id,session,task,id,id_dataset,x_train,y_train,x_test,y_test,params,apply_feature_selection=params["feature_selection"])
                edit_trained_model(session_id,id,id_dataset,task,target,int(params["n_iter"]))
                i=i+1
                
        elif task=="regression":

            i=0
            for id in model_id:
                params=param_dict[i]
                regression_model_param_check(user_id,session,task,id,id_dataset,x_train,y_train,x_test,y_test,params,apply_feature_selection=params["feature_selection"])
                edit_trained_model(session_id,id,id_dataset,task,target,int(params["n_iter"]))
                i=i+1
        elif task=="clustering":
            i=0
            for id in model_id:
                params=param_dict[i]
                clustering_model_param_check(user_id,session,task,id,id_dataset,x,params,apply_feature_selection=params["feature_selection"])
                edit_trained_model(session_id,id,id_dataset,task,target,int(params["n_iter"]))
                i=i+1

        shutil.rmtree(f"{session}")
        change_train_session_status(session_id,"completed")
        save_split_strategy(id_dataset,split_strategy["train_percentage"]*100,split_strategy["test_percentage"]*100,split_strategy["seed"],split_strategy["shuffle"])
    except Exception as e:  
        self.update_state(state='FAILURE', meta={'exception': str(e)})
        change_train_session_status(session_id,"Failed")
        if os.path.exists(str(session_id)):
            shutil.rmtree(str(session_id))

        return e