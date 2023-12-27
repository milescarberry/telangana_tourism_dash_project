import numpy as np

import pandas as pd

# import matplotlib.pyplot as plt

# import seaborn as sns

# sns.set_context("paper", font_scale = 1.4)

# from plotly import express as exp, graph_objects as go, io as pio

# pio.templates.default == 'ggplot2'

# from plotly.subplots import make_subplots

import pickle

import datetime as dt

from pandas_utils.pandas_utils_2 import *

# import ipywidgets

# from IPython.display import display

import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)




years = [str(i) for i in range(2016, 2020, 1)]



domestic_visitors_loc = "./input_files/domestic_visitors/domestic_visitors_"

foreign_visitors_loc = "./input_files/foreign_visitors/foreign_visitors_"


dom_df = pd.DataFrame()


for_df = pd.DataFrame()



for y in years:
    
    
    dom_df = pd.concat([dom_df, pd.read_csv(domestic_visitors_loc + y + ".csv")], axis = 0)

    
    for_df = pd.concat([for_df, pd.read_csv(foreign_visitors_loc + y + ".csv")], axis = 0)
    





def get_quarter(date_vals):
    
    
    quarts = []
    
    
    for d in date_vals:
        
        
        
        if int(dt.datetime.strftime(d, "%m")) <= 3:
            
            
            quarts.append("Q1")
            
        
        elif int(dt.datetime.strftime(d, "%m")) > 3 and int(dt.datetime.strftime(d, "%m")) <= 6:
            
            
            quarts.append("Q2")
            
        
        
        elif int(dt.datetime.strftime(d, "%m")) > 6 and int(dt.datetime.strftime(d, "%m")) <= 9:
            
            
            
            quarts.append("Q3")
            
            
        
        
        elif int(dt.datetime.strftime(d, "%m")) > 9 and int(dt.datetime.strftime(d, "%m")) <= 12:
            
            
            quarts.append("Q4")
            
        
        
        else:
            
            
            quarts.append(np.nan)
    
    
    
    return quarts
            
            
            
        
        
        
    
    


def clean_df(df):
    
    
#     df['date'] = df['date'].apply(lambda x: dt.datetime.strptime(x, "%d-%m-%Y"))
    
    
    df['month_year'] = df['month'] +  ' ' + df['year'].apply(lambda x: str(x))
    
    
    df['date'] = df['month_year'].apply(lambda x: dt.datetime.strptime(x, "%B %Y"))
    
    
#     df['month_year'] = df['month_year'].apply(lambda x: dt.datetime.strptime(x, "%B %Y"))
    
    
    df['quarter'] = get_quarter(df['date'].map(dt.datetime.date).values.tolist())
    
    
    df['year_quarter'] = df['year'].apply(lambda x: str(x)) + '-' + df['quarter'] 
    
    
    df['visitors'] = df.visitors.replace(' ', 'nan')
    
    
#     try:
        
#         df['visitors'] = df.visitors.apply(lambda x: int(x) if x != ' ' else 0)
        
    
#     except BaseException as e:
        
        
    df['visitors'] = df.visitors.apply(lambda x: int(x) if str(x).strip().lower() != 'nan' else 0)
        
        
#         print(f"\n\n{e}\n\n")
        
    
#     else:
        
#         pass
    
    
    
    df = df[['district', 'date', 'month', 'year', 'month_year', 'quarter', 'year_quarter', 'visitors']]
    
    
    return df









dom_df = clean_df(dom_df)

for_df = clean_df(for_df)


dom_df = dom_df.sort_values(by = ['district', 'date'], ascending = [True, True])

for_df = for_df.sort_values(by = ['district', 'date'], ascending = [True, True])




dom_df['foreign_visitors'] = for_df.visitors



dom_df = dom_df.rename({"visitors": 'domestic_visitors'}, axis = 1)



dom_df['d_to_f_ratio'] = dom_df.domestic_visitors / dom_df.foreign_visitors





# dom_df.d_to_f_ratio = dom_df.d_to_f_ratio.replace(np.inf, 0)





dom_df['district'] = dom_df['district'].apply(lambda x: x.strip())


dom_df = dom_df.replace("Warangal (Rural)", "Warangal")


dom_df = dom_df.replace("Warangal (Urban)", "Hanumakonda")


dom_df = dom_df.replace("Jayashankar Bhoopalpally", "Jayashankar Bhupalpally")


dom_df = dom_df.replace("Komaram Bheem Asifabad", "Kumurambheem Asifabad")


dom_df = dom_df.replace("Mahbubnagar", "Mahabubnagar")


dom_df['district'] = dom_df['district'].replace("Medchal", "Medchal Malkajgiri")


dom_df['district'] = dom_df['district'].replace("Ranga Reddy", "Rangareddy")


dom_df['district'] = dom_df['district'].replace("Yadadri Bhongir", "Yadadri Bhuvanagiri")






dom_df.to_parquet("./cleaned_data/dom_df.parquet")

