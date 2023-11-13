import re
import pandas as pd
import numpy as np 
import joblib
import plotly.express as px
import plotly.io as pio
import plotly.io as pio
pio.renderers.default = 'iframe'
pio.templates.default = "plotly_white"

def codex_PI(session,model, data, criterion, nreps, relative=False, variable_level=True, show_reps=False):
    original_feature_names=joblib.load(f"{session}/original_cols_list.joblib")
    x_data = data[0]
    y_data = data[1]
    original_pred = model.predict(x_data)
    original_error = criterion(y_data, original_pred)

    all_results = pd.DataFrame([])
    for rep in range(nreps):
        results = []
        total = 0
        for predictor in x_data:
            x_data_copy = x_data.copy()
            x_data_copy[predictor] = x_data[predictor].sample(frac=1).values
            perbutated_pred = model.predict(x_data_copy)
            perbutated_error = criterion(y_data, perbutated_pred)
            PI_value = original_error-perbutated_error 
            if relative:
                total += PI_value
            results.append({'features': predictor, 'score': PI_value})
        resultsdf = pd.DataFrame(results).sort_values(by = 'score', ascending = False)
        if show_reps:
            resultsdf['rep'] = rep + 1
        if relative:
            resultsdf['score'] = resultsdf['score']/resultsdf['score'].sum()
        if variable_level:
            dummy_importance_features = resultsdf['features']
            out = pd.DataFrame([])
            for feature in original_feature_names:
                feat_idx = np.array([re.match(f'^{feature}_', col) is not None for col in dummy_importance_features])
                feat_idx2 = dummy_importance_features[feat_idx].tolist()
                if not feat_idx.any():
                    t1 = pd.DataFrame({'variables': [feature], "features": [feature]})
                    out = pd.concat([out, t1])
                else:
                    t2 = pd.DataFrame({'variables': [feature]*len(feat_idx2), "features": feat_idx2})
                    out = pd.concat([out, t2])
            resultsdf = resultsdf.merge(out, on=['features'], how='left')
        all_results = pd.concat([all_results, resultsdf])
    return all_results

def generate_quantiles(df, probs = [0.025, 0.5, 0.975], type="features"):
    if len(probs) != 3:
        raise ValueError('Specify, lower, middle and upper percentiles in probs')
    df = df[[type, 'score']]
    result = df.drop_duplicates(type)
    result = result.drop(['score'], axis=1)
    for p in probs:
        out = df.groupby(type, as_index=False).quantile(q=p)
        out.columns = [type, 'percentile_'+str(p)]
        result = result.merge(out, on=type, how='left')
    result = result.sort_values(by=['percentile_0.5'], ascending=True)
    result.columns = [type, 'lower', 'value', 'upper']
    return result

def generate_mean(df, abs=True, dropzero=True, type="features"):
    '''
        df: generated variable importance data frame
        type: either features or variables
    '''
    df = df.groupby(type, as_index=False).mean()
    df['sign'] = np.where(df['score']>=0, "+", "-")
    if abs:
        df['score'] = np.abs(df['score'])
    df = df.sort_values(by = 'score', ascending = True)
    if dropzero:
        df = df[df['score'] != 0]
    return df

def generate_permutation_summary(df, probs=[0.025, 0.5, 0.975], abs=True, dropzero=False, type="features"):
    if 'rep' in df.columns:
        df = df.drop(['rep'], axis=1)
    out_quant = generate_quantiles(df, probs=probs, type=type)
    out_mean = generate_mean(df, abs=abs, dropzero=dropzero, type=type)
    out = {'PI_quant': out_quant, 'PI_mean': out_mean, 'type': type}
    return out

def plot_PI(df, df_type="mean", type="features"):
    if df_type == "mean":
        df = df['PI_mean'].copy()
        num_data_points = len(df)
        adaptive_height = max(600, num_data_points * 40 ) 
        df['score'] = df['score']*df.loc[:,"sign"].replace(['-',"+"],[-1,1])
        df=df.sort_values(by='score', ascending=True)
        fig = px.scatter(
            df, x='score', y=df[type], color=df['sign'],
            color_discrete_sequence=['purple','blue'], height=adaptive_height, width=1000
            , error_x=df['score'] - df['score']
            , error_x_minus=df['score']
        )
        fig.update_layout(showlegend=False, yaxis_title=None, xaxis_title="Score")
    elif df_type == "quant":
        df = df['PI_quant'].copy()
        num_data_points = len(df)
        adaptive_height = max(600, num_data_points * 40 ) 
        fig = px.scatter(y = df[type]
            , x = df['value']
            , error_x=df['upper'] - df['value']
            , error_x_minus=df['value'] - df['lower']
            , color_discrete_sequence=['gray']
            ,height=adaptive_height
            ,width=1000
        )
        fig.update_layout(yaxis_title=None, xaxis_title="Score")
    return fig