import os
from urllib.parse import urlparse
import pyreadstat
import pandas as pd

## EXTEND: add new file formats to the list
supported_files = ['csv', 'excel', 'json', "stata", "spss", "sas"]

class SupportedDataFormats:
    """
    Upload various file formats suported by pandas
    
    Parameters
    ----------
    file_path: str
        path (or an url) to the file
    extension:
        file extension. 
        
    Returns
    -------
    path: str
    extension: boolean
    """
    def __init__(self, file_path, extension):
        self.path = file_path
        self.file_name=os.path.basename(urlparse(self.path).path)
        self.extension = self.path.lower().endswith(extension)
        
    def get_file_name(self):
        return self.file_name
        
    def csv(self):
        df = pd.read_csv(self.path,low_memory=False)
        return df

    def excel(self):
        df = pd.read_excel(self.path)
        return df
    
    def json(self):
        df = pd.read_json(self.path)
        return df
    
    def stata(self):
        df = pd.read_stata(self.path)
        return df
    
    def spss(self):
        df = pd.read_spss(self.path)
        return df
    
    def sas(self):
        # df = pd.read_sas(self.path, chunksize=None, format="sas7bdat")
        df, meta = pyreadstat.read_sas7bdat(self.path)
        return df
    
    
    ## EXTEND: Add other file formats bellow:
    # def newextension(self):
    #    df = ...
    #    return df
        
def read_supported(file_path, supported_files=supported_files, file_name=False):
    """
    Upload various file formats suported by pandas
    
    Parameters
    ----------
    file_path: str
        path (or an url) to the file
    supported_files:
        A list of supported files. These include 'csv', 'excel', 'json', 
        "stata", "spss" and "sas"
        
    Returns
    -------
    DataFrame:
        a pandas data frame
    
    """
    for file in supported_files:
        if file=="stata":
            file="dta"
        if file=="spss":
            file="sav"
        if file=="sas":
            file="sas7bdat"
        if file=="excel":
            file="xlsx"
            
        out = SupportedDataFormats(file_path, file)
        
        if file_name:
            return out.file_name
        else:
            if out.extension:
                if file=="csv":
                    return out.csv()
                elif file=="xlsx":
                    return out.excel()
                elif file=="json":
                    return out.json()
                elif file=="dta":
                    return out.stata()
                elif file=="sav":
                    return out.spss()
                elif file=="sas7bdat":
                    return out.sas()
                ## EXTEND: replace newextension below
                # elif file==newextension:
                #    return out.newextension()
                else:
                    return "File format not supported"

