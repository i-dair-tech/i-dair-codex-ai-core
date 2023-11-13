from django.test import TestCase

# Create your tests here.
import sys
import os 
import pandas as pd
import json
import pandas_profiling
from pandas_profiling.utils.cache import cache_file
# Get the current working directory
path=os.getcwd()
# Get the parent directory of the current working directory
dataset_path_prefix = os.path.dirname(path)+"/"
project_path = os.path.join(path,'tools')

sys.path.insert(0, project_path)
from db_tools import get_dataset
from plots_tools import generate_plots
from pandas_profiling_cleaner import clean
from suppoeted_data_formats import read_supported

from django.test import TestCase
from django.urls import reverse

class MyAppTestCase(TestCase):
    
    def setUp(self):
        self.file_path,self.filename=get_dataset(1)
        self.file_path=dataset_path_prefix+self.file_path
        print("setUp")
        
    def tearDown(self):
        print("tear down")
    
    def test_get_dataset(self):
        # df = pd.read_csv(file_path): CHANGED: Mar 6, 2023
        df = read_supported(self.file_path) #  CHANGED: Mar 6, 2023 
        print(f"df type {type(df)} , df shape {df.shape} ")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(self.file_path,str)
        self.assertIsInstance(self.filename, str)
        self.assertTrue(len(self.file_path)>0)
        self.assertTrue(len(self.filename)>0)
        
    def test_clean_data(self):
        REPORT_PATH="hdp_data_Analysis.json"
        # df = pd.read_csv(file_path): CHANGED: Mar 6, 2023
        df = read_supported(self.file_path) #  CHANGED: Mar 6, 2023
        profile_report = df.profile_report(html={"style": {"full_width": True}},title="hdp_data_Analysis")
        profile_report.to_file(REPORT_PATH)
        with open(REPORT_PATH) as f:
            data = json.load(f)
        os.remove(REPORT_PATH)
        # Return the cleaned data as a JSON response
        clean_data=clean(data,df) 
        self.assertIsInstance(clean_data,dict)
