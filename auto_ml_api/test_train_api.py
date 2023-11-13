import unittest
import json
import sys 
import os 
from django.http import HttpRequest, QueryDict
from django.test import TestCase,RequestFactory

# Get the current working directory
path=os.getcwd()
# Get the parent directory of the current working directory
dataset_path_prefix = os.path.dirname(path)+"/"

# Construct the path to the tools and training folders
train_path = os.path.join(path,'auto_ml_api/train')

# Add the paths to the tools and training folders to the module search path
sys.path.insert(0, train_path)

import train_api
class TestTrainApi(unittest.TestCase):
    
    def setUp(self):
        self.factory = RequestFactory()
        
    def tearDown(self):
        print("tear down")
        
    def test_train(self):
        
        kmeans_params={
            "idDataset":1,
            "train_percentage":70,
            "test_percentage":30,
            "shuffle":True,
            "task":"clustering",
            "seed":43,
            "target":"",

            "models":[
            {   "id_model":8,
                "n_iter":10
            }]
            }
        kmeans_params_fail={
            "idDataset":1,
            "train_percentage":70,
            "test_percentage":30,
            "shuffle":True,
            "seed":43,
            "target":"",

            "models":[
            {   "id_model":8,
                "n_iter":10
            }]
            }
        classification_params={
            "idDataset":1,
            "train_percentage":70,
            "test_percentage":30,
            "seed":42,
            "shuffle":True,
            "task":"classification",
            "target":"Married",
            "models":[
            {"id_model":1,"max_iter":10,"random_state":10,"n_iter":10},
            {"id_model":4,"n_iter":10,"max_iter":12}]
            
            }
        classification_params_fail={
            "idDataset":1,
            "train_percentage":70,
            "test_percentage":30,
            "seed":42,
            "shuffle":True,
            "target":"Married",
            "models":[
            {"id_model":1,"max_iter":10,"random_state":10,"n_iter":10},
            {"id_model":4,"n_iter":10,"max_iter":12}]
            
            }
        regression_params={
            "idDataset":1,
            "train_percentage":70,
            "test_percentage":30,
            "shuffle":True,
            "task":"regression",
            "seed":43,
            "target":"remission",
            "models":[
            {   "id_model":5,
                "n_iter":10
            }]
            }
        regression_params_fail={
            "idDataset":1,
            "train_percentage":70,
            "test_percentage":30,
            "shuffle":True,
            "seed":43,
            "target":"remission",
            "models":[
            {   "id_model":5,
                "n_iter":10
            }]
            }
        params=[
            (kmeans_params,"success"),
            (kmeans_params_fail,"failure"),
            (classification_params,"success"),
            (classification_params_fail,"failure"),
            (regression_params,"success"),
            (regression_params_fail,"failure")
            ]
        for param , status in params:
            if status =="success":
                # Convert the dictionary to a QueryDict object
                query_dict = QueryDict.fromkeys(param)

                # Create a request object
                request = HttpRequest()
                request.POST = query_dict
                
                request = self.factory.post('/train/', data=json.dumps(param), content_type='application/json')
                request_body = json.loads(request.body.decode())
                print("request type ",type(request_body))
                print("request value ", request_body)
                response = train_api.train(request)
                self.assertEqual(response.status_code,200)
                response=json.loads(response.content.decode())
                print("respnse value",response)
                self.assertEqual(response["success"], True)
                print("response type", type(response))
            else:
                # Convert the dictionary to a QueryDict object
                query_dict = QueryDict.fromkeys(param)

                # Create a request object
                request = HttpRequest()
                request.POST = query_dict
                
                request = self.factory.post('/train/', data=json.dumps(param), content_type='application/json')
                request_body = json.loads(request.body.decode())
                print("request type ",type(request_body))
                print("request value ", request_body)
                response = train_api.train(request)
                self.assertEqual(response.status_code,400)
                response=json.loads(response.content.decode())
                print("respnse value",response)
                self.assertEqual(response["success"], False)
                print("response type", type(response))
                
if __name__ == '__main__':
    unittest.main()