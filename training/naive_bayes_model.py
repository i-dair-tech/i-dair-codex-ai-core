import mlflow.sklearn
from skopt import BayesSearchCV
from set_decimals import set_decimals
from sklearn.naive_bayes import GaussianNB
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from roc_curve_generator import generate_roc_curve
import numpy as np
import copy
import os 
import sys
import pandas as pd
import glob
import plotly.io as pio
import plotly.graph_objects as go
import joblib
import shutil
import json
from custom_permutation_importance import codex_PI,generate_permutation_summary, plot_PI
# Get the current working directory
path=os.getcwd()
# Get the parent directory of the current working directory
dataset_path_prefix = os.path.dirname(path)+"/"
tools_path = os.path.join(dataset_path_prefix,'tools')
sys.path.insert(0, tools_path)

from db_tools import add_trained_model, epochs_tracker 
from bootsrapfuns import bootstrap
from custom_exceptions import CustomException
from set_mlflow_tracking_uri import set_tracking_uri
import logging

logger = logging.getLogger("main")

epoch=1
def on_step (optim_result):
    global id_model 
    global id_dataset
    global epoch
    global session_id
    epochs_tracker(session_id,id_model,id_dataset,epoch)
    print(f'Epoch {epoch}')  # print the current epoch
    epoch+=1
    if epoch > iterations:
        epoch=0
    
id_model=None
id_dataset=None 
iterations=None
session_id=None

def naive_bayes(user_id,session,model_id, dataset_id, x_train, y_train, x_test, y_test, n_iter, optimizer="BayesianOptimization",param_space=None,task="classification"):
    global id_model 
    global id_dataset
    global iterations
    global session_id
    session_id=int(session)
    iterations = copy.deepcopy(n_iter)
    id_model = copy.deepcopy(model_id)
    id_dataset = copy.deepcopy(dataset_id)
    scores={"precision":precision_score,"recall": recall_score, "f1_score":f1_score,"accuracy": accuracy_score}
    # Define the number of permutations for the test
    nperms = 100
    
    folds_number=param_space.get("folds_number",None)
    if folds_number:
        del param_space["folds_number"]
    
    if folds_number:
        cross_validation_method="StratifiedKFold"
    else:
        cross_validation_method="3-fold cross validation"
    # Define the model and the hyperparameter search space
    #model = ComplementNB()
    model=GaussianNB()
    print(f"param space value {param_space}\n param space type {type(param_space)}")

    joblib.dump(param_space,f"{session}/{model_id}/param_space.joblib")
    
    # Convert the Series to a NumPy array
    y_train_new = y_train.values
    # Get the number of classes in the data
    num_classes = len(np.unique(y_train_new))
    print(f" y train unique values {np.unique(y_train_new)} , num clasess {num_classes}")
    # Set the average parameter based on the number of classes
    if num_classes == 2:
        average = 'binary'
    else:
        average = 'macro'
    print(f"average {average}")
    
    # set Tracking Uri
    set_tracking_uri() 
    
    # Create the experiment
    timestamp = datetime.now()
    model_name="Naive Bayes"
    experiment_name = f'{model_name} Experiment ' + str(timestamp)
    experiment_id = mlflow.create_experiment(experiment_name)
  
    if optimizer == "RandomSearch":
        # Create the random search object
        search = RandomizedSearchCV(model, param_space, n_iter=n_iter, cv=folds_number)
    elif optimizer == "BayesianOptimization":
        # Create the Bayesian optimization search object
        search = BayesSearchCV(model, param_space, n_iter=n_iter, cv=folds_number, random_state=123,
                               optimizer_kwargs={'base_estimator': 'GP', 'acq_func': 'EI'},
                               n_points=100)
        
    
    
    # Start an mlflow run
    with mlflow.start_run(experiment_id=experiment_id) as run:
        print("mlflow")
        try:
            run_id = run.info.run_id
            logger.info(f"train launched run_id:{run_id} in task/session: {session_id}",extra={'user_id': user_id})
            add_trained_model(run_id,id_model,id_dataset,0,session)
            # Fit the model to the training data
            search.fit(x_train, y_train,callback=on_step)

        except Exception as e:
            # Log the error message
            mlflow.log_param('error_message', str(e))
            raise CustomException(f"{model_id}-Possible issue with Hyperparameter combination or search: {e}")
        else:
            # Get the best model from BayesSearchCV
            best_model = search.best_estimator_

            # Make predictions on the test data
            predictions = best_model.predict(x_test)
            
            # Calculate the permutation feature importance
            importances=codex_PI(session,model=best_model, data=[x_test,y_test], criterion=accuracy_score, nreps=100)
            importances = importances.reset_index(drop=True)
            
            importances_summary=generate_permutation_summary(importances,type="variables")
            
            PI_mean_variables = plot_PI(importances_summary, df_type='mean', type="variables")
            
            pio.write_json(PI_mean_variables, f'{session}/{model_id}/importance_fig.json')
            
            importances.to_json(f"{session}/{model_id}/importance_scores.json",orient='columns')

            # Bootsrap model callibration
            bootstrap_estimates = bootstrap(model=best_model, X=x_test, y=y_test, metrics=scores,average=average, task=task, model_name=model_name, nreps=nperms)
            bootstrap_estimates.to_csv(f"{session}/{model_id}/bootstrap_estimates.csv",index=False)
            mlflow.log_artifact(f'{session}/{model_id}/bootstrap_estimates.csv')
            
            data_preprocessing_files = glob.glob(f"{session}/*.joblib")
            for file in data_preprocessing_files:
                if file.find("feature_engineering_tracking.joblib")==-1: 
                    mlflow.log_artifact(file)
                    
            feature_engineering_files = glob.glob(f"{session}/{model_id}/*.joblib")
            for file in feature_engineering_files:
                mlflow.log_artifact(file)                
                
            mlflow.log_artifact(f"{session}/{model_id}/importance_fig.json")   
            mlflow.log_artifact(f"{session}/{model_id}/importance_scores.json")

            # Evaluate the model's performance
            accuracy = search.score(x_test, y_test)
            precision = precision_score(y_test, predictions,average=average)
            recall = recall_score(y_test, predictions,average=average)
            f1 = f1_score(y_test, predictions,average=average)

            # Log the evaluation metrics
            mlflow.log_metric('accuracy', set_decimals(accuracy))
            mlflow.log_metric('precision', set_decimals(precision))
            mlflow.log_metric('recall', set_decimals(recall))
            mlflow.log_metric('f1_score', set_decimals(f1))
            roc_curve_fig=generate_roc_curve(best_model,x_test,y_test,model_name)
            roc_curve_list=[roc_curve_fig,True]
            roc_curve_fig_path=f"{session}/{model_id}/roc_curve.joblib"
            joblib.dump(roc_curve_list,roc_curve_fig_path)
            mlflow.log_artifact(roc_curve_fig_path)      
                        
            #log optimizer
            mlflow.log_param("optimizer",optimizer)
            
            #log cross validation method
            mlflow.log_param("cross_validation_method",cross_validation_method)
            
            # Log iterations
            mlflow.log_param("iterations",n_iter)

            #log the number of folds
            mlflow.log_param("folds_number",folds_number)
            
            # Log the best hyperparameters
            mlflow.log_params(search.best_params_)

            # Log the model
            registered_model_name="naive_bayes_model_"+str(timestamp)
            mlflow.sklearn.log_model(best_model, 'model', registered_model_name=registered_model_name)
            shutil.rmtree(f"{session}/{model_id}")
            # Update epoch status
            epochs_tracker(session_id,id_model,id_dataset,n_iter,"completed")
            global epoch
            epoch=1    