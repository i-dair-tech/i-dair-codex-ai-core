from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import ExtraTreeClassifier, DecisionTreeRegressor
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from skopt import BayesSearchCV
import pandas as pd
import numpy as np
import copy
from skopt.space import Categorical
class ValidateTrain:
    def __init__(self,hyperparameters,model_id,task,generated_data_rows=50,n_iter=20):
        self.folds_number=hyperparameters.get("folds_number",None)
        self.n_iter=n_iter
        self.num_rows=generated_data_rows
        self.task=task
        
        del hyperparameters["n_iter"]   
        del hyperparameters["parameter_search"]
        del hyperparameters["folds_number"]
        
        hyperparameters_dict_copy=copy.deepcopy(hyperparameters)
        
        for key,val in hyperparameters_dict_copy.items():
            if len(val)==0:
                del hyperparameters[key]
                
        self.hyperparameters=hyperparameters
        self.model_id=model_id
        
        if self.model_id == 1 and self.hyperparameters["penalty"]==[None] :
            self.hyperparameters["penalty"]=["none"]
        
        for key , value in self.hyperparameters.items():
            self.hyperparameters[key]=self.unique_elements(value)
        if "random_state" in hyperparameters:
            self.hyperparameters["random_state"] = Categorical(hyperparameters["random_state"] )

        self.model=None
        self.x=None
        self.y=None
        self.search=None
        self.get_model()
        self.generate_data()
        self.param_search_algorithm()
        
    def get_model(self):
        
        if self.model_id == 1:
            self.model= LogisticRegression() 
                       
        elif self.model_id == 2:    
            self.model= GaussianNB()
            
        elif self.model_id==3:
            self.model= ExtraTreeClassifier()  
            
        elif self.model_id==4:
            self.model=SVC()
            
        elif self.model_id==5:
            self.model=LinearRegression()
            
        elif self.model_id==6:
            self.model=XGBClassifier()
            
        elif self.model_id==7:
            self.model=XGBRegressor(importance_type="gain")
            
        elif self.model_id==8:
            self.model= KMeans() 
        
        elif self.model_id==9:
            self.model= RandomForestClassifier() 
            
        elif self.model_id==10:
            self.model=RandomForestRegressor()
        
        elif self.model_id==11:
            self.model=MLPClassifier(early_stopping=True) 
        
        elif self.model_id==12:
            self.model= DecisionTreeRegressor() 

    def unique_elements(self, lst):
        seen = set()  # A set to store unique elements
        result = []   # A list to store the final unique elements

        for item in lst:
            # If the item is a float or can be converted to a float with equivalent integer value
            if isinstance(item, (float, int)) and int(item) == item and (not isinstance(item, (bool))) :
                item = int(item)  # Convert to integer to treat 2.0 as 2
            if item not in seen:
                seen.add(item)
                result.append(item)

        return result

    def generate_data(self):

        # Create a DataFrame with columns A, B, C
        data = {
            'A': np.random.rand(self.num_rows),
            'B': np.random.rand(self.num_rows),
            'C': np.random.rand(self.num_rows),
        }
        if self.task=="classification":
            # For binary classification, create a 'class' column with random 0s and 1s
            data['class'] = np.random.randint(0, 2, size=self.num_rows)
        else:
            # For regression, create a 'class' column with random float values between 0 and 50
            data['class'] = np.random.uniform(0, 50, size=self.num_rows)
        
        # Create the DataFrame
        df = pd.DataFrame(data)    
        self.x=df.drop("class", axis=1)
        self.y=df["class"]
         
    def param_search_algorithm(self):
        self.search = BayesSearchCV(self.model, self.hyperparameters, n_iter=self.n_iter, cv=self.folds_number, random_state=42)
        
    def validate (self):
        try:
            self.search.fit(self.x,self.y)
        except Exception as e:
            return False ,e
        return True , None
        
        

