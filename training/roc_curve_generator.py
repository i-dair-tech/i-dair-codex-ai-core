import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score

def generate_roc_curve(model,X_test,y_test,model_name):
  
    # Predict probabilities for the positive class (class 1)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    # Calculate the AUC (Area Under the Curve) for ROC
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    # Create an ROC curve plot using Plotly
    fig = go.Figure()

    hover_text=['Threshold: {:.2f}'.format(threshold) for threshold in thresholds]

    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=model_name+' (AUC={:.2f})'.format(roc_auc),text=hover_text))


    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate'),
    )
    return fig
