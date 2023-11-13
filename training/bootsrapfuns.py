import numpy as np
import pandas as pd
import random

def bootstrap(model, X, y, metrics,average=None,task="", model_name="", nreps=100, random_state=911):
    random.seed(random_state)
    check_x = check_y = False
    if isinstance(X, pd.DataFrame):
        check_x = True
    if isinstance(y, pd.DataFrame):
        check_y = True
    bootstrap_results=[]
    
    for n in range(nreps):
        id_pick = np.random.choice(np.shape(X)[0], size=(np.shape(X)[0]), replace=True)
        if check_y:
            y = y.iloc[id_pick]
        else:
            y = y[id_pick]
        if check_x:
            X = X.iloc[id_pick]
        else:
            X = X[id_pick, :]
        predictions = model.predict(X)
        for metric_name, metric in metrics.items():
            if metric_name=="rmse":
                score = metric(y, predictions)
                score=np.sqrt(score)
            else:
                if task=="classification" and metric_name!="accuracy":
                    score = metric(y, predictions,average=average)
                else:
                    score = metric(y, predictions)
            result={"resample":n+1,"model":model_name,"metric":metric_name,"score":score}
            bootstrap_results.append(result)
    df=pd.DataFrame(bootstrap_results)
    metric_quantiles = df.groupby(['metric'])['score'].quantile([0.25, 0.5, 0.75]).unstack()
    metric_quantiles.columns = ['lower', 'median', 'upper']
    df = df.merge(metric_quantiles, left_on='metric', right_index=True)
    return df

