from training.univariate_selection import univariate_selection
from training.recursive_feature_elimination import recursive_feature_elimination
from principal_component_analysis import pca_func

def feature_selection(session,model_id,x_train,task,model_name=None,y_train=None,x_test=None,model=None):
    if task!="clustering":
        if model_name=="Multilayer perceptron":
            x_train_selected,x_test_selected = univariate_selection(session,model_id,model,x_train,y_train,x_test)
            return x_train_selected ,x_test_selected            
        else:
            x_train_selected,x_test_selected = recursive_feature_elimination(session,model_id,model,x_train,y_train,x_test,task)
            return x_train_selected ,x_test_selected
    
    elif task=="clustering":
        x_train_selected=pca_func(session,model_id,x_train) 
        return x_train_selected
    