import numpy as np
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import cross_val_score
import joblib
import pandas as pd 

def univariate_selection(session,model_id,model,x_train,y_train,x_test):
    feature_engineering_tracking_dict=joblib.load(f"{session}/feature_engineering_tracking.joblib")
    feature_engineering_tracking_dict["feature_selection"]="univariate selection"
    
    # Create a list to store the cross-validation scores
    cv_scores = []
    k_values=[i for i in range(0,(min(x_train.shape)+1)) if i>0]
    for k in k_values:
        # Create a SelectKBest object with the specified value of K
        bestfeatures = SelectKBest(score_func=chi2, k=k)
        select = bestfeatures.fit(x_train,y_train)
        x_train_selected = select.transform(x_train)
        # Use cross-validation to evaluate the performance of the model with the selected features
        scores= cross_val_score(model,x_train_selected, y_train, cv=5)
        # Append the mean cross-validation score to the list of scores
        cv_scores.append(np.mean(scores))
        
    best_index = np.argmax(cv_scores)
    best_k = k_values[best_index]
    feature_engineering_tracking_dict["selected_features"]=best_k
    print(f"Highest mean accuracy {np.max(cv_scores):.2f} achieved at K={best_k}")
    bestfeatures = SelectKBest(score_func=chi2, k=best_k)
    select = bestfeatures.fit(x_train,y_train)
    joblib.dump(select, f'{session}/{model_id}/univariate_selection.joblib')
    x_train_selected = select.transform(x_train)
    x_test_selected = select.transform(x_test)
    feature_names = x_train.columns[select.get_support()]
    x_train_selected_df = pd.DataFrame(x_train_selected, columns=feature_names)
    x_test_selected_df = pd.DataFrame(x_test_selected, columns=feature_names)
    feature_engineering_tracking_dict["selected_features_names"]=x_train_selected_df.columns.tolist()
    joblib.dump(feature_engineering_tracking_dict,f"{session}/{model_id}/feature_engineering_tracking.joblib")
    return x_train_selected_df, x_test_selected_df