import pandas as pd
import plotly.express as px
import numpy as np
import os 
import plotly.io as pio
import json
import re
import plotly.graph_objects as go

def is_roman_number(num):

    pattern = re.compile(r"""
                                ^M{0,3}
                                (CM|CD|D?C{0,3})?
                                (XC|XL|L?X{0,3})?
                                (IX|IV|V?I{0,3})?$
            """, re.VERBOSE)

    if re.match(pattern, num):
        return True
    
    return False

def roman_to_int(s):
    
    roman = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000,'IV':4,'IX':9,'XL':40,'XC':90,'CD':400,'CM':900}
    i = 0
    num = 0
    while i < len(s):
        if i+1<len(s) and s[i:i+2] in roman:
            num+=roman[s[i:i+2]]
            i+=2
        else:
            #print(i)
            num+=roman[s[i]]
            i+=1
    return s,num 

def generate_plots(df, col, plot_type,num=False):
           
    # Create a figure using Plotly Express
    if plot_type == "histogram" and num==False:

        unique_values=df[col].unique().tolist()
        is_number_list=[]
        is_roman_list=[]
        for val in unique_values:
            val=str(val)
            is_roman_list.append(is_roman_number(val))
            is_number_list.append(val.isnumeric())
        is_number=all(is_number_list)
        is_roman=all(is_roman_list)
        
        df[col] = df[col].apply(str)
        
        # Select the specified column from the DataFrame
        df = df[col]
        
        if is_roman: 
            roman_list=[]     
            for val in unique_values:
                roman,num= roman_to_int(val)
                roman_list.append((roman,num))
                
            sorted_roman_list = sorted(roman_list, key=lambda x: x[1])
            sorted_roman_list=[roman for roman , _ in sorted_roman_list ]
            fig = px.histogram(df, x=col, labels={'x':col, 'y':'frequency'}, category_orders={col:sorted_roman_list},text_auto=True)
            
        elif is_number:
            unique_values=sorted(unique_values) 
            unique_values=[str(i) for i in unique_values ]
            fig = px.histogram(df, x=col, labels={'x':col, 'y':'frequency'}, category_orders={col:unique_values},text_auto=True)
            
        else:
            fig = px.histogram(df, x=col, labels={'x':col, 'y':'frequency'},text_auto=True).update_xaxes(categoryorder='category ascending')
        
    elif plot_type =="histogram" and num==True:
        
        # Select the specified column from the DataFrame
        df = df[col]
        # Calculate the optimal bins using the numpy.histogram_bin_edges() function
        bins = np.histogram_bin_edges(df, bins='fd')
        counts, bins = np.histogram(df, bins=bins)
        bins = 0.5 * (bins[:-1] + bins[1:])
        fig = px.bar(x=bins, y=counts, labels={'x':col, 'y':'frequency'},text_auto=True)
        
    
    # Add interactive features to the figure, such as filtering and zooming
    fig.update_layout(
          dragmode="select",
          hovermode="closest",
          selectdirection="h"
    )
    
    json_string = pio.to_json(fig,pretty=True)
    json_string = json.loads(json_string)
    return json_string

def figures_unifier(figures):

    # Create a combined figure
    combined_fig = go.Figure()
    combined_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Random'))
    for fig in figures:
        # Add the logistic regression ROC curve
        combined_fig.add_trace(fig['data'][0])
    combined_fig.update_layout(
        title='Combined Receiver Operating Characteristic (ROC) Curves',
        title_x=0.5,
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate'),
    )

    return combined_fig