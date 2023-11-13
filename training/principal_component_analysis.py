import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import pandas as pd 
import joblib
def pca_func(session,model_id,x):
    feature_engineering_tracking_dict=joblib.load(f"{session}/feature_engineering_tracking.joblib")
    feature_engineering_tracking_dict["feature_selection"]="principal component analysis"
    var_explained=[0.75, 0.8, 0.9, 0.95, 1]
    cv_scores, n_components_ = [], []
    for v in var_explained:
        pca = PCA(n_components=v)
        cv_scores.append(np.mean(cross_val_score(pca, x)))
        pca.fit(x)
        n_components_.append(pca.n_components_)
        
    ### Optimal components
    best_components = get_best_pcs(cv_scores, n_components_)
    print("%0.0f features selected out of %0.0f" % (best_components, x.shape[1]))
    feature_engineering_tracking_dict["selected_features"]=x.shape[1]
    joblib.dump(feature_engineering_tracking_dict,f"{session}/{model_id}/feature_engineering_tracking.joblib")
    ### Perform PCA with optimal PCs and return PC df
    pca = PCA(n_components=best_components)
    pca.fit(x)
    pca_scores_df = pca.transform(x)
    top_features = np.abs(pca.components_)[0]
    top_features_idx = np.argsort(top_features)[::-1][:best_components]
    top_features = x.columns[top_features_idx].values
    top_features = top_features.reshape(-1, best_components)
    pca_scores_df=pd.DataFrame(pca_scores_df, columns=top_features[0])
    return pca_scores_df

def get_best_pcs(cv_scores, componets):
    best_index = np.argmax(cv_scores)
    return componets[best_index]