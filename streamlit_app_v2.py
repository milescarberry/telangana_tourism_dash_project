#!/usr/bin/python -tt

import numpy as np

import pandas as pd

# import matplotlib.pyplot as plt

# import seaborn as sns

# sns.set_context("paper", font_scale = 1.4)

from plotly import express as exp, graph_objects as go, io as pio

from plotly.subplots import make_subplots

import pickle

import datetime as dt

from pandas_utils.pandas_utils_2 import *


from subprocess import call


import streamlit as st


import warnings




st.set_page_config(


        page_title = "Page Title", 


        layout = 'wide'


        )



pio.templates.default = 'ggplot2'


warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)






@st.cache_data
def get_dataset():


        query_list = ["python", "data_preprocessing.py"]


        call(query_list)


        df = pd.read_parquet("./cleaned_data/dom_df.parquet")


        return df




# Dashboard Title


st.write(

        "<h1><center>Title</center></h1>",


        unsafe_allow_html = True


        )




st.write("<br><br>", unsafe_allow_html = True)



# Some More Text


st.write(

        "<h5><center>Some more text comes here.</center></h5>", 

        unsafe_allow_html = True

        )



st.write("<br><br>", unsafe_allow_html = True)




# Getting Data


dom_df = get_dataset()




# Sidebar Filters




with st.sidebar:


        # Filters Title


        st.write("<h1><center>Filters</center></h1><br>", unsafe_allow_html = True)


        # District Filter


        district_filter = ['All']
            
            
        unique_districts = list(dom_df.district.unique())
            
            
        unique_districts.sort()
            
            
        district_filter.extend(unique_districts)


        district_filt = st.multiselect(


                "Select Districts: ", 

                district_filter, 

                default = 'All'

                )


        st.write("<br>", unsafe_allow_html = True)



        # Time Axis Filter



        if 'time_ax' not in st.session_state:


                st.session_state.time_ax = 'Month'




        time_axes = ['Month', 'Quarter']



        st.session_state.time_ax = st.selectbox(


                "Select Time Axis: ", 

                time_axes, 

                index = 0

                )


        st.write("<br>", unsafe_allow_html = True)




        # Metric Filter




        metrics = [
            
            'Domestic Visitors', 
            
            'Foreign Visitors', 
            
            'Domestic and Foreign Visitors', 
            
            'Domestic to Foreign Visitor Ratio'

        ]


        metric = st.selectbox(


                "Select Metric: ", 

                metrics, 

                index = 0

        )


        st.write("<br>", unsafe_allow_html = True)



        # Calculation Filter



        calcs = {

                "Month": [

                    'Percent Change From Previous Month', 

                    'YOY', 

                    'YTM'

                ],


                "Quarter": [

                    "Percent Change From Previous Quarter", 

                    'YOY', 

                    'YTQ'

                ]

            }





        calc = st.selectbox(


                        "Select Calculation: ", 

                        calcs[st.session_state.time_ax], 

                        index = 0

                )









# Basic Stats Chart






def plot_stats(visitor_type = metric, district_filter = district_filt, time_axis = st.session_state.time_ax):
        
        
    
    df_ = dom_df
    
    
    
    if 'All' in district_filter:
        
        
        district_filter = list(df_.district.unique())
        
    
    else:
        
        
        district_filter = district_filter
        
    
    
    
    district_filter.sort()
        
    
    
    
    fig_df = df_[df_.district.isin(district_filter)]
    
    
    
    
    if visitor_type == 'Domestic and Foreign Visitors':
        

        
        
        if time_axis == 'Month':
            

            fig_df = fig_df.groupby(

                ['date'], 

                as_index = False, 

                dropna = False

            ).agg(

                {

                "foreign_visitors": pd.Series.sum, 

                 "domestic_visitors": pd.Series.sum

                }


            )
            
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])


#             fig = go.Figure()


            fig.add_trace(


                go.Scatter(


                    x = fig_df['date'],


                    y = fig_df['domestic_visitors'],


                    mode = 'lines+markers',


                    name = 'Domestic'


                    ),
                
                
                
                secondary_y = False




            )



            fig.add_trace(
                

                go.Scatter(


                    x = fig_df['date'],


                    y = fig_df['foreign_visitors'],


                    mode = 'lines+markers',


                    name = 'Foreign'




                ),
                
                
                
                secondary_y = True



            )
        
        
        
        else:
            
            
            fig_df = fig_df.groupby(

                ['year_quarter'], 

                as_index = False, 

                dropna = False

            ).agg(

                {

                "foreign_visitors": pd.Series.sum, 

                 "domestic_visitors": pd.Series.sum

                }


            )

            
            
#             fig = go.Figure()


            fig = make_subplots(specs=[[{"secondary_y": True}]])
    


            fig.add_trace(


                go.Scatter(


                    x = fig_df['year_quarter'],


                    y = fig_df['domestic_visitors'],


                    mode = 'lines+markers',


                    name = 'Domestic'


                    ),
                
                
                
                secondary_y = False




            )



            fig.add_trace(

                go.Scatter(


                    x = fig_df['year_quarter'],


                    y = fig_df['foreign_visitors'],


                    mode = 'lines+markers',


                    name = 'Foreign'




                ),
                
                
                secondary_y = True



            )

        
    
        
        
        fig.update_yaxes(title = "Domestic Visitors", secondary_y = False)
        
        
        
        fig.update_yaxes(title = "Foreign Visitors", secondary_y = True)
        
        
        
        fig.update_yaxes(secondary_y=True, showgrid=False)



        fig.update_xaxes(title = ' ', tickangle=-45)
        
        
        
        hovertemp = "<br><br>".join([
            
            "<b>%{x}</b>", 
            
            "<b>%{y:.2s} visitors</b><extra></extra>"
        
        
        ])

        
        
        fig.update_traces(hovertemplate = hovertemp)


                

        
        
        
        if len(district_filter) == 33:
        
        
            fig.update_layout(
                
                title = dict(
                    
                    text = f"Domestic vs. Foreign Visitors for All Districts",

                    pad = dict(b = 110, l = 150, r = 50 )
                
                )
            
            )
        
        
        else:
            
                   
            fig.update_layout(
                
                title = dict(
                    
                    text = f"Domestic vs. Foreign Visitors for {st.session_state.time_ax.title()} For {', '.join(district_filter[:-1:1]) + ' & ' + district_filter[-1] if len(district_filter) <= 3 else 'The Selected'} Districts",


                    pad = dict(b = 110, l = 150, r = 50 )
                
                )
            
            )
            
        
        
#         fig.update_layout(width = 780)
        
        
        st.plotly_chart(fig, use_container_width = True)
            
            



        
    
    elif visitor_type == 'Domestic Visitors':
        
        
        
        if time_axis == 'Month':
            
            
        

            fig_df = fig_df.groupby(

                ['date'], 

                as_index = False, 

                dropna = False

            ).agg(

                {

    #             "foreign_visitors": pd.Series.sum, 

                 "domestic_visitors": pd.Series.sum

                }


            )


            fig = exp.line(

                fig_df, 

                x='date', 

                y='domestic_visitors', 

                labels={'x':' ', 'y':'Domestic Visitors'},

                markers = True,

                title=''


            )
        
        
        
        else:
            
            
                    
            fig_df = fig_df.groupby(

                ['year_quarter'], 

                as_index = False, 

                dropna = False

            ).agg(

                {

    #             "foreign_visitors": pd.Series.sum, 

                 "domestic_visitors": pd.Series.sum

                }


            )


            fig = exp.line(

                fig_df, 

                x='year_quarter', 

                y='domestic_visitors', 

                labels={'x':' ', 'y':'Domestic Visitors'},

                markers = True,

                title=''


            )





        fig.update_yaxes(title = "Domestic Visitors")



        fig.update_xaxes(title = ' ', tickangle=-45)
            
            
        
        
        
        hovertemp = "<br><br>".join([
            
            
            "<b>%{x}</b>", 
            
            
            "<b>%{y:.2s} visitors</b><extra></extra>"
        
        
        ])
        
        
        fig.update_traces(hovertemplate = hovertemp)



            
            
        
        
        if len(district_filter) == 33:
        
        
            fig.update_layout(
                
                title = dict(
                    
                    text = f"Domestic Visitors for All Districts",


                    pad = dict(b = 110, l = 150, r = 50 )
                
                )
            
            )
            
        
            
        
        
        else:
            
                   
            fig.update_layout(
                
                title = dict(
                    
                    text = f"Domestic Visitors for {st.session_state.time_ax.title()} For {', '.join(district_filter[:-1:1]) + ' & ' + district_filter[-1] if len(district_filter) <= 3 else 'The Selected'} Districts",


                    pad = dict(b = 110, l = 150, r = 50 )
                
                )
            
            )
            
            
        
#         fig.update_layout(width = 780)
            
        
        
        
        st.plotly_chart(fig, use_container_width = True)
            
    
        
        
    
    
    elif visitor_type == 'Foreign Visitors':
        
        
        if time_axis == 'Month':
        
        
            fig_df = fig_df.groupby(

                ['date'], 

                as_index = False, 

                dropna = False

            ).agg(

                {

                "foreign_visitors": pd.Series.sum

    #              "domestic_visitors": pd.Series.sum

                }


            )



            fig = exp.line(

                fig_df, 

                x='date', 

                y='foreign_visitors', 

                labels={'x':' ', 'y':'Foreign Visitors'},

                markers = True,

                title=''


            )

        
        
        else:
            
            
            
            fig_df = fig_df.groupby(

                ['year_quarter'], 

                as_index = False, 

                dropna = False

            ).agg(

                {

                "foreign_visitors": pd.Series.sum

    #              "domestic_visitors": pd.Series.sum

                }


            )



            fig = exp.line(

                fig_df, 

                x='year_quarter', 

                y='foreign_visitors', 

                labels={'x':' ', 'y':'Foreign Visitors'},

                markers = True,

                title=''


            )



        fig.update_yaxes(title = "Foreign Visitors")



        fig.update_xaxes(title = ' ', tickangle=-45)
            
        
        
                
        hovertemp = "<br><br>".join([
            
            
            "<b>%{x}</b>", 
            
            
            "<b>%{y:.2s} visitors</b><extra></extra>"
        
        
        ])
        
        
        fig.update_traces(hovertemplate = hovertemp)


            
            
        
        
        
        if len(district_filter) == 33:
        
        
            fig.update_layout(
                
                title = dict(
                    
                    text = f"Foreign Visitors for All Districts",


                    pad = dict(b = 110, l = 150, r = 50 )
                
                )
            
            )
        
        
        
        else:
            
                   
            fig.update_layout(
                

                title = dict(

                    
                    text = f"Foreign Visitors for {st.session_state.time_ax.title()} For {', '.join(district_filter[:-1:1]) + ' & ' + district_filter[-1] if len(district_filter) <= 3 else 'The Selected'} Districts",


                    pad = dict(b = 110, l = 150, r = 50 )
                
                )
            
            )
            
            
            
            
        
#         fig.update_layout(width = 780)
            
            
        
        
        st.plotly_chart(fig, use_container_width = True)
        
    
    
    
    else:
        
        
        
        if time_axis == 'Month':
        
        
            fig_df = fig_df.groupby(

                ['date'], 

                as_index = False, 

                dropna = False

            ).agg(

                {

                "d_to_f_ratio": pd.Series.mean

    #              "domestic_visitors": pd.Series.sum

                }


            )
            
            
            
            fig_df['d_to_f_ratio'] = fig_df['d_to_f_ratio'].replace(np.inf, 0)

            


            fig = exp.line(

                fig_df, 

                x='date', 

                y="d_to_f_ratio", 

                labels={'x':' ', 'y':'d_to_f_ratio'},

                markers = True,

                title=''


            )

        
        
        else:
            
            
            
            fig_df = fig_df.groupby(

                ['year_quarter'], 

                as_index = False, 

                dropna = False

            ).agg(

                {

                "d_to_f_ratio": pd.Series.mean

    #              "domestic_visitors": pd.Series.sum

                }


            )
            
            
            
            fig_df['d_to_f_ratio'] = fig_df['d_to_f_ratio'].replace(np.inf, 0)



            fig = exp.line(

                fig_df, 

                x='year_quarter', 

                y='d_to_f_ratio', 

                labels={'x':' ', 'y':'d_to_f_ratio'},

                markers = True,

                title=''


            )




        fig.update_yaxes(title = "D to F Ratio")



        fig.update_xaxes(title = ' ', tickangle=-45)
            
        
        
        
                
        hovertemp = "<br><br>".join([
            
            
            "<b>%{x}</b>", 
            
            
            "<b>D to F Ratio: %{y:.2s}</b><extra></extra>"
        
        
        ])
        
        
        fig.update_traces(hovertemplate = hovertemp)


            
            
        
        
        
        if len(district_filter) == 33:
        
        
            fig.update_layout(
                
                title = dict(
                    
                    text = f"Domestic to Foreign Visitor Ratio for All Districts",


                    pad = dict(b = 110, l = 150, r = 50 )
                
                )
            
            )
        
        
        
        else:
            
                   
            fig.update_layout(
                
                title = dict(
                    
                    text = f"Domestic to Foreign Visitor Ratio for {st.session_state.time_ax.title()} For {', '.join(district_filter[:-1:1]) + ' & ' + district_filter[-1] if len(district_filter) <= 3 else 'The Selected'} Districts",


                    pad = dict(b = 110, l = 150, r = 50 )
                
                )
            
            )
            
            
            
            
        
#         fig.update_layout(width = 780)
            
            
        
        
        st.plotly_chart(fig, use_container_width = True)





plot_stats()
        
        
        
            
            
# Calc Chart



# Calc Funcs



# Percent Difference Func





def percent_change_from_previous(df, col = 'month', metric = 'domestic_visitors', district_filter = []):
    
    
    time_val = ''
    
    
    if col == 'month':
        
        
        
        time_val = 'date'
        
    
    
    else:
        
        
        time_val = 'year_quarter'
        
        
        df['date'] = df.year_quarter.apply(lambda x: dom_df[dom_df.year_quarter == x]['date'].min())
        
        
        
    
    
    if 'list' not in str(type(metric)).lower():
        
    
    
        df[f'prev_{metric}'] = df[metric].shift(1)


        df[f'{metric}_percent_change'] = (

            (df[metric] - df[f'prev_{metric}']) / df[f'prev_{metric}']

        ) * 100
        
        
        
        
        
        fig = exp.line(

                df, 

                x=time_val, 

                y=f'{metric}_percent_change', 

                labels={'x':' ', 'y':f'{metric}_percent_change'},

                markers = True,

                title=''


            )






        fig.update_yaxes(title = f'{metric}_percent_change'.replace("_", ' ').title())
        
        
   
        
        
        hovertemp = "<br><br>".join([
            
            
            "<b>%{x}</b>", 
            
            
            "<b>%{y:.2s}%</b>" + f" <b>vs. Previous {st.session_state.time_ax}</b><extra></extra>"
        
        
        ])
        
        
        fig.update_traces(hovertemplate = hovertemp)




        if time_val == 'year_quarter':


                                fig.update_xaxes(title = ' ', tickangle = -45)


                else:

                                fig.update_xaxes(title = ' ')



            
            
        
        
        if len(district_filter) == 33:
        
        
            fig.update_layout(
                
                title = dict(
                    
                    text = f"{metric.replace('_', ' ').title()} Percent Change From Previous {st.session_state.time_ax.title()} For All Districts",


                    pad = dict(b = 110, l = 150, r = 50 )
                
                )
            
            )
            
        
            
        
        
        else:
            
                   
            fig.update_layout(
                
                title = dict(
                    
                    text = f"{metric.replace('_', ' ').title()} Percent Change From Previous {st.session_state.time_ax.title()} For {st.session_state.time_ax.title()} For {', '.join(district_filter[:-1:1]) + ' & ' + district_filter[-1] if len(district_filter) <= 3 else 'The Selected'} Districts",


                    pad = dict(b = 110, l = 150, r = 50 )
                
                )
            
            )
            
            
        
#         fig.update_layout(width = 780)
            
        
        
        
        st.plotly_chart(fig, use_container_width = True)
            
        
        
        
        
        
        
        
    
    
    else:
        
        
        
        for m in metric:

            df[f'prev_{m}'] = df[m].shift(1)

            df[f'{m}_percent_change'] = (

                (df[m] - df[f'prev_{m}']) / df[f'prev_{m}']

            ) * 100


        fig = make_subplots(specs=[[{"secondary_y": True}]])


        fig.add_trace(


            go.Scatter(


                x=df[time_val],


                y=df[f'{metric[0]}_percent_change'],


                mode='lines+markers',


                name='Domestic'


            ),



            secondary_y=False




         )

            

        fig.add_trace(


            go.Scatter(


                x=df[time_val],


                y=df[f'{metric[1]}_percent_change'],


                mode='lines+markers',


                name='Foreign'




            ),



            secondary_y=True



        )



        fig.update_yaxes(title = "Domestic Visitors", secondary_y = False, showgrid = True)
        
        
        
        fig.update_yaxes(title = "Foreign Visitors", secondary_y = True, showgrid = False)
        


        if time_val == 'year_quarter':


                        fig.update_xaxes(title = ' ', tickangle = -45)


                else:

                        fig.update_xaxes(title = ' ')
        
        


        hovertemp = "<br><br>".join([
            

            "<b>%{x}</b>", 

            
            "<b>%{y:.2s}%</b>" + f" <b>vs. Previous {st.session_state.time_ax}</b><extra></extra>"
        
        
        ])
        
        
        fig.update_traces(hovertemplate = hovertemp)


        
        
        
        if len(district_filter) == 33:
        
        
            fig.update_layout(
                
                title = dict(
                    
                    text = f"{metric[0].replace('_', ' ').title()} vs. {metric[1].replace('_', ' ').title()} Percent Change From Previous {st.session_state.time_ax.title()} For All Districts",


                    pad = dict(b = 110, l = 150, r = 50 )
                
                )
            
            )
            
        
            
        
        
        else:
            
                   
            fig.update_layout(
                
                title = dict(
                    
                    text = f"{metric[0].replace('_', ' ').title()} vs. {metric[1].replace('_', ' ').title()} Percent Change From Previous {st.session_state.time_ax.title()} For {', '.join(district_filter[:-1:1]) + ' & ' + district_filter[-1] if len(district_filter) <= 3 else 'The Selected'} Districts",


                    pad = dict(b = 110, l = 150, r = 50 )
                
                )
            
            )
        

        
        st.plotly_chart(fig, use_container_width = True)







# YOY Func




def yoy_calc(df, col = 'month', metric = 'domestic_visitors', district_filter = []):
    
    
    time_val = ''
    
    
    
    if col == 'month':
        
        
        time_val = 'date'
        
    
    
    else:
        
        time_val = 'year_quarter'
        
        
        df['date'] = df.year_quarter.apply(lambda x: dom_df[dom_df.year_quarter == x]['date'].min())
        
        
        
    
    
    
    if 'list' not in str(type(metric)).lower():
        
        
        
        df['prev_year_date'] = df['date'].apply(lambda x: x - dt.timedelta(365))
        
        
        
        prev_year_dates = df.prev_year_date.values
        
        
        prev_year_vals = []
        
        
        for d in prev_year_dates:
    
    
            vals = df[df['date'] == d][metric].values
        
        
            if len(vals) == 0:
                
                
                prev_year_vals.append(np.nan)
                
            
            else:
                
                
                prev_year_vals.append(vals[0])
                
        
        
        
        df[f'prev_year_{metric}'] = prev_year_vals
        
        
        
        
        df[f'{metric}_prev_year_percent_change'] = (
            
            
            (df[metric] - df[f'prev_year_{metric}']) / df[f'prev_year_{metric}']
        
        
        ) * 100
        
       
        
        
                
        fig = exp.line(

                df, 

                x=time_val, 

                y=f'{metric}_prev_year_percent_change', 

                labels={'x':' ', 'y':f'{metric}_prev_year_percent_change'},

                markers = True,

                title=''


            )
       

        
        if time_val == 'year_quarter':


                        fig.update_xaxes(title = ' ', tickangle = -45)


                else:

                        fig.update_xaxes(title = ' ')


        
        fig.update_yaxes(title = f'{metric}_prev_year_percent_change'.replace("_", " ").title())



        hovertemp = "<br><br>".join([
            
            
            "<b>%{x}</b>", 
            
            
            "<b>%{y:.2f} %</b>" + " <b>vs. Previous Year</b><extra></extra>"
        
        
        ])
        
        
        
        fig.update_traces(hovertemplate = hovertemp)



                







            
            
        
        
        
        if len(district_filter) == 33:
        
        
            fig.update_layout(
                
                title = dict(
                    
                    text = f"{metric.replace('_', ' ').title()} YOY Percent Change For All Districts",


                    pad = dict(b = 110, l = 150, r = 50 )
                
                )
            
            )
            
        
            
        
        else:
            
                   
            fig.update_layout(
                
                title = dict(
                    
                    text = f"{metric.replace('_', ' ').title()} YOY Percent Change For {st.session_state.time_ax.title()} For {', '.join(district_filter[:-1:1]) + ' & ' + district_filter[-1] if len(district_filter) <= 3 else 'The Selected'} Districts",


                    pad = dict(b = 110, l = 150, r = 50 )
                
                )
            
            )
            
            
        
#         fig.update_layout(width = 780)
            
        
        
        
        st.plotly_chart(fig, use_container_width = True)
        
        
        
        
        
        
    
    
    else:
        
        
        
        df['prev_year_date'] = df['date'].apply(lambda x: x - dt.timedelta(365))
        
        
        
        prev_year_dates = df.prev_year_date.values
        
        
        
        
        # Dual Axis
        
        
        for m in metric:
            
            
            
            prev_year_vals = []

            

            for d in prev_year_dates:


                vals = df[df['date'] == d][m].values


                if len(vals) == 0:


                    prev_year_vals.append(np.nan)


                else:


                    prev_year_vals.append(vals[0])




            df[f'prev_year_{m}'] = prev_year_vals
            
            

            df[f'{m}_prev_year_percent_change'] = (


                (df[m] - df[f'prev_year_{m}']) / df[f'prev_year_{m}']


            ) * 100




            
            
            
            # Dual Axis Chart
            
            
            
        fig = make_subplots(specs=[[{"secondary_y": True}]])




        fig.add_trace(


            go.Scatter(


                x = df[time_val],


                y = df[f'{metric[0]}_prev_year_percent_change'],


                mode = 'lines+markers',


                name = 'Domestic'


            ),
                
                
                
            secondary_y = False




        )



        fig.add_trace(
                

                go.Scatter(


                    x = df[time_val],


                    y = df[f'{metric[1]}_prev_year_percent_change'],


                    mode = 'lines+markers',


                    name = 'Foreign'




            ),
                
                
                
            secondary_y = True



        )
            
            
            
        fig.update_yaxes(title = f"{metric[0].replace('_', ' ').title()} YOY % Change", secondary_y = False, showgrid = True)
        
        
        
        fig.update_yaxes(title = f"{metric[1].replace('_', ' ').title()} YOY % Change", secondary_y = True, showgrid = False)
        


        if time_val == 'year_quarter':


                        fig.update_xaxes(title = ' ', tickangle = -45)


                else:

                        fig.update_xaxes(title = ' ')
        
        
        
        hovertemp = "<br><br>".join([
            
            "<b>%{x}</b>", 
            
            "<b>%{y:.2f} % vs. Previous Year</b><extra></extra>"
        
        
        ])
        
        
        
        fig.update_traces(hovertemplate = hovertemp)


        
        
        
        
                
        if len(district_filter) == 33:
        
        
            fig.update_layout(
                
                title = dict(
                    
                    text = f"{metric[0].replace('_', ' ').title()} vs. {metric[1].replace('_', ' ').title()} YOY Percent Change For All Districts",


                    pad = dict(b = 110, l = 150, r = 50 )
                
                )
            
            )
            
        
            
        
        
        else:
            
                   
            fig.update_layout(
                
                title = dict(
                    
                    text = f"{metric[0].replace('_', ' ').title()} vs. {metric[1].replace('_', ' ').title()} YOY Percent Change For {st.session_state.time_ax.title()} For {', '.join(district_filter[:-1:1]) + ' & ' + district_filter[-1] if len(district_filter) <= 3 else 'The Selected'} Districts",


                    pad = dict(b = 110, l = 150, r = 50 )
                
                )
            
            )
            
        
        
        
        
        st.plotly_chart(fig, use_container_width = True)
    
    
    


        

# YTM / YTQ Func



def ytm_ytq_calc(df, col = 'month', metric = 'domestic_visitors', district_filter = []):
    
   

    time_val = ''
    
    
    
    
    if col == 'month':
        
        
        time_val = 'date'
        
    
    
    else:
        
        time_val = 'year_quarter'
        
        
        df['date'] = df.year_quarter.apply(lambda x: dom_df[dom_df.year_quarter == x]['date'].min())
        
        
    
    
    df['first_date_year'] = df['date'].apply(
        
        lambda x: x - dt.timedelta(int(dt.datetime.strftime(x, "%j")) - 1)
    
    
    )
    
    
        
    
    
    
    if 'list' not in str(type(metric)).lower():
        
        
        
        
        first_date_year_dates = df.first_date_year.values
        
        
        first_date_year_vals = []
        
        
        for d in first_date_year_dates:
    
    
            vals = df[df['date'] == d][metric].values
        
        
            if len(vals) == 0:
                
                
                first_date_year_vals.append(np.nan)
                
            
            else:
                
                
                first_date_year_vals.append(vals[0])
                
        
        
        
        df[f'first_date_year_{metric}'] = first_date_year_vals
        
        
        
        
        df[f'{metric}_first_date_year_percent_change'] = (
            
            
            (df[metric] - df[f'first_date_year_{metric}']) / df[f'first_date_year_{metric}']
        
        
        ) * 100
        
        
        
        
        
                
        fig = exp.line(

                df, 

                x=time_val, 

                y=f'{metric}_first_date_year_percent_change', 

                labels={'x':' ', 'y':f'{metric}_first_date_year_percent_change'},

                markers = True,

                title=''


            )




        fig.update_yaxes(title = f'{metric}_first_date_year_percent_change'.replace('_', ' ').title())
        

        
        if time_val == 'year_quarter':


                        fig.update_xaxes(title = ' ', tickangle = -45)


                else:

                        fig.update_xaxes(title = ' ')


        
        
        if time_val == 'date':
        
        
            hovertemp = "<br><br>".join([


                "<b>%{x}</b>", 


                " <b>YTM: </b>" + "<b>%{y:.2f} %</b><extra></extra>"


            ])
        
        
        
        else:
            
            
                    
            hovertemp = "<br><br>".join([


                "<b>%{x}</b>", 


                " <b>YTM: </b>" + "<b>%{y:.2f} %</b><extra></extra>"


            ])
        
        
        
        fig.update_traces(hovertemplate = hovertemp)




            
            
        if time_val == 'date':
            
            
        
        
            if len(district_filter) == 33:


                fig.update_layout(

                    title = dict(

                        text = f"{metric.replace('_', ' ').title()} YTM Percent Change For All Districts",


                        pad = dict(b = 110, l = 150, r = 50 )

                    )

                )




            else:


                fig.update_layout(

                    title = dict(

                        text = f"{metric.replace('_', ' ').title()} YTM Percent Change For {st.session_state.time_ax.title()} For {', '.join(district_filter[:-1:1]) + ' & ' + district_filter[-1] if len(district_filter) <= 3 else 'The Selected'} Districts",


                        pad = dict(b = 110, l = 150, r = 50 )

                    )

                )

        
        
        
        
        else:
            
            
            
            if len(district_filter) == 33:


                fig.update_layout(

                    title = dict(

                        text = f"{metric.replace('_', ' ').title()} YTQ Percent Change For All Districts",


                        pad = dict(b = 110, l = 150, r = 50 )

                    )

                )




            else:


                fig.update_layout(

                    title = dict(

                        text = f"{metric.replace('_', ' ').title()} YTQ Percent Change For {st.session_state.time_ax.title()} For {', '.join(district_filter[:-1:1]) + ' & ' + district_filter[-1] if len(district_filter) <= 3 else 'The Selected'} Districts",


                        pad = dict(b = 110, l = 150, r = 50 )

                    )

                )

            
            
            
            
            
            
        
#         fig.update_layout(width = 780)
            
        
        
        
        st.plotly_chart(fig, use_container_width = True)
        
        
        
        
        
        
    
    
    else:
        
        
        
        
        # Dual Axis
        
        
        for m in metric:
            
            
            
            first_date_year_dates = df.first_date_year.values
        

            first_date_year_vals = []


            for d in first_date_year_dates:


                vals = df[df['date'] == d][m].values


                if len(vals) == 0:


                    first_date_year_vals.append(np.nan)


                else:


                    first_date_year_vals.append(vals[0])




            df[f'first_date_year_{m}'] = first_date_year_vals




            df[f'{m}_first_date_year_percent_change'] = (


                (df[m] - df[f'first_date_year_{m}']) / df[f'first_date_year_{m}']


            ) * 100







            
        # Dual Axis Chart
            
            
            
        fig = make_subplots(specs=[[{"secondary_y": True}]])




        fig.add_trace(


                go.Scatter(


                    x = df[time_val],


                    y = df[f'{metric[0]}_first_date_year_percent_change'],


                    mode = 'lines+markers',


                    name = 'Domestic'


                    ),
                
                
                
                secondary_y = False




        )



        fig.add_trace(
                

                go.Scatter(


                    x = df[time_val],


                    y = df[f'{metric[1]}_first_date_year_percent_change'],


                    mode = 'lines+markers',


                    name = 'Foreign'




                ),
                
                
                
                secondary_y = True



        )

            
            
        if time_val == 'date':
            
            
            
            fig.update_yaxes(title = f"{metric[0].replace('_', ' ').title()} YTM % Change", secondary_y = False, showgrid = True)



            fig.update_yaxes(title = f"{metric[1].replace('_', ' ').title()} YTM % Change", secondary_y = True, showgrid = False)

            
            
            hovertemp = "<br><br>".join([
            
                "<b>%{x}</b>", 

                "<b>YTM: </b>" + "<b>%{y:.2f} %</b><extra></extra>"
        
        
            ])
            
            

        else:
        
        
             fig.update_yaxes(title = f"{metric[0].replace('_', ' ').title()} YTM % Change", secondary_y = False, showgrid = True)
        
        
        
             fig.update_yaxes(title = f"{metric[1].replace('_', ' ').title()} YTM % Change", secondary_y = True, showgrid = False)
        
        

             if time_val == 'year_quarter':


                                        fig.update_xaxes(title = ' ', tickangle = -45)


                         else:

                                        fig.update_xaxes(title = ' ')

            

             hovertemp = "<br><br>".join([
            
                "<b>%{x}</b>", 

                "<b>YTQ: </b>" + "<b>%{y:.2f} %</b><extra></extra>"
        
        
             ])
        
        

        
        
        
        fig.update_traces(hovertemplate = hovertemp)

        
        
        
        
        if time_val == 'date':
            
            

            if len(district_filter) == 33:


                fig.update_layout(

                    title = dict(

                        text = f"{metric[0].replace('_', ' ').title()} vs. {metric[1].replace('_', ' ').title()} YTM Percent Change For All Districts",


                        pad = dict(b = 110, l = 150, r = 50 )

                    )

                )





            else:


                fig.update_layout(

                    title = dict(

                        text = f"{metric[0].replace('_', ' ').title()} vs. {metric[1].replace('_', ' ').title()} YTM Percent Change For {st.session_state.time_ax.title()} For {', '.join(district_filter[:-1:1]) + ' & ' + district_filter[-1] if len(district_filter) <= 3 else 'The Selected'} Districts",


                        pad = dict(b = 110, l = 150, r = 50 )

                    )

                )

    
            
            
        
        
        else:
            
        
        
                
            if len(district_filter) == 33:


                fig.update_layout(

                    title = dict(

                        text = f"{metric[0].replace('_', ' ').title()} vs. {metric[1].replace('_', ' ').title()} YTQ Percent Change For All Districts",


                        pad = dict(b = 110, l = 150, r = 50 )

                    )

                )





            else:


                fig.update_layout(

                    title = dict(

                        text = f"{metric[0].replace('_', ' ').title()} vs. {metric[1].replace('_', ' ').title()} YTQ Percent Change For {st.session_state.time_ax.title()} For {', '.join(district_filter[:-1:1]) + ' & ' + district_filter[-1] if len(district_filter) <= 3 else 'The Selected'} Districts",


                        pad = dict(b = 110, l = 150, r = 50 )

                    )

                )



                
        
        st.plotly_chart(fig, use_container_width = True)
    
    

            
            
# The Calcs Chart




def more_calcs(
    
    
    metric = metric, 
    
    
    time_axis = st.session_state.time_ax, 
    
    
    calc = calc,
    
    
    district_filter = district_filt

):
    
    

    if district_filter == 'All':

        

        district_filter = unique_districts



    else:


        district_filter = district_filter




    df_ = dom_df



    df_ = df_[df_.district.isin(district_filter)]




    if time_axis == 'Month':



        if metric == 'Domestic Visitors':



            df_ = df_.groupby(

                ['date'], 

                dropna = False, 

                as_index = False

            ).agg(


                {"domestic_visitors": pd.Series.sum}

            )


            if calc == 'Percent Change From Previous Month':



                percent_change_from_previous(

                    df = df_, 


                    col = 'month', 


                    metric = 'domestic_visitors', 


                    district_filter = district_filter


                )




            elif calc == 'YOY':


                yoy_calc(

                    df = df_, 

                    col = 'month', 

                    metric = 'domestic_visitors', 

                    district_filter = district_filter

                )



            elif calc == 'YTM':


                ytm_ytq_calc(

                    df = df_, 

                    col = 'month', 

                    metric = 'domestic_visitors', 

                    district_filter = district_filter

                )



            else:

                pass





        elif metric == 'Foreign Visitors':



            df_ = df_.groupby(

                ['date'], 

                dropna = False, 

                as_index = False

            ).agg(


                {"foreign_visitors": pd.Series.sum}

            )



            if calc == 'Percent Change From Previous Month':



                percent_change_from_previous(

                    df = df_, 


                    col = 'month', 


                    metric = 'foreign_visitors', 


                    district_filter = district_filter


                )




            elif calc == 'YOY':


                yoy_calc(

                    df = df_, 

                    col = 'month', 

                    metric = 'foreign_visitors', 

                    district_filter = district_filter


                )



            elif calc == 'YTM':


                ytm_ytq_calc(

                    df = df_, 

                    col = 'month', 

                    metric = 'foreign_visitors', 

                    district_filter = district_filter

                )



            else:

                pass




        elif metric == 'Domestic and Foreign Visitors':



            df_ = df_.groupby(

                ['date'], 

                dropna = False, 

                as_index = False

            ).agg(


                {

                    "domestic_visitors": pd.Series.sum, 


                    "foreign_visitors": pd.Series.sum

                }

            )



            if calc == 'Percent Change From Previous Month':



                percent_change_from_previous(

                    df = df_, 


                    col = 'month', 


                    metric = ['domestic_visitors', 'foreign_visitors'], 


                    district_filter = district_filter


                )




            elif calc == 'YOY':


                yoy_calc(

                    df = df_, 

                    col = 'month', 

                    metric = ['domestic_visitors', 'foreign_visitors'], 

                    district_filter = district_filter

                )



            elif calc == 'YTM':



                ytm_ytq_calc(

                    df = df_, 

                    col = 'month', 

                    metric = ['domestic_visitors', 'foreign_visitors'], 

                    district_filter = district_filter

                )



            else:

                pass




        elif metric == 'Domestic to Foreign Visitor Ratio':



            df_ = df_.groupby(

                ['date'], 

                dropna = False, 

                as_index = False

            ).agg(

                {"d_to_f_ratio": pd.Series.mean}

            )


            if calc == 'Percent Change From Previous Month':



                percent_change_from_previous(

                    df = df_, 


                    col = 'month', 


                    metric = 'd_to_f_ratio', 


                    district_filter = district_filter


                )




            elif calc == 'YOY':


                yoy_calc(

                    df = df_, 

                    col = 'month', 

                    metric = 'd_to_f_ratio', 

                    district_filter = district_filter

                )



            elif calc == 'YTM':


                ytm_ytq_calc(

                    df = df_, 

                    col = 'month', 

                    metric = 'd_to_f_ratio', 

                    district_filter = district_filter

                )



            else:

                pass



        else:


            pass






    else:


        if metric == 'Domestic Visitors':



            df_ = df_.groupby(

                ['year_quarter'], 

                dropna = False, 

                as_index = False

            ).agg(


                {"domestic_visitors": pd.Series.sum}

            )



            if calc == 'Percent Change From Previous Quarter':



                percent_change_from_previous(

                    df = df_, 


                    col = 'quarter', 


                    metric = 'domestic_visitors', 


                    district_filter = district_filter


                )




            elif calc == 'YOY':


                yoy_calc(

                    df = df_, 

                    col = 'quarter', 

                    metric = 'domestic_visitors', 

                    district_filter = district_filter


                )



            elif calc == 'YTQ':


                ytm_ytq_calc(

                    df = df_, 

                    col = 'quarter', 

                    metric = 'domestic_visitors', 

                    district_filter = district_filter

                )



            else:

                pass




        elif metric == 'Foreign Visitors':



            df_ = df_.groupby(

                ['year_quarter'], 

                dropna = False, 

                as_index = False

            ).agg(


                {"foreign_visitors": pd.Series.sum}

            )



            if calc == 'Percent Change From Previous Quarter':



                percent_change_from_previous(

                    df = df_, 


                    col = 'quarter', 


                    metric = 'foreign_visitors', 


                    district_filter = district_filter


                )




            elif calc == 'YOY':


                yoy_calc(

                    df = df_, 

                    col = 'quarter', 

                    metric = 'foreign_visitors', 

                    district_filter = district_filter


                )



            elif calc == 'YTQ':


                ytm_ytq_calc(

                    df = df_, 

                    col = 'quarter', 

                    metric = 'foreign_visitors', 

                    district_filter = district_filter

                )



            else:

                pass




        elif metric == 'Domestic and Foreign Visitors':



            df_ = df_.groupby(

                ['year_quarter'], 

                dropna = False, 

                as_index = False

            ).agg(


                {

                    "domestic_visitors": pd.Series.sum, 


                    "foreign_visitors": pd.Series.sum

                }

            )



            if calc == 'Percent Change From Previous Quarter':



                percent_change_from_previous(

                    df = df_, 


                    col = 'quarter', 


                    metric = ['domestic_visitors', 'foreign_visitors'], 


                    district_filter = district_filter


                )




            elif calc == 'YOY':


                yoy_calc(

                    df = df_, 

                    col = 'quarter', 

                    metric = ['domestic_visitors', 'foreign_visitors'], 

                    district_filter = district_filter


                )



            elif calc == 'YTQ':


                ytm_ytq_calc(

                    df = df_, 

                    col = 'quarter', 

                    metric = ['domestic_visitors', 'foreign_visitors'], 

                    district_filter = district_filter

                )



            else:

                pass




        elif metric == 'Domestic to Foreign Visitor Ratio':



            df_ = df_.groupby(

                ['year_quarter'], 

                dropna = False, 

                as_index = False

            ).agg(

                {"d_to_f_ratio": pd.Series.mean}

            )


            if calc == 'Percent Change From Previous Quarter':



                percent_change_from_previous(

                    df = df_, 


                    col = 'quarter', 


                    metric = 'd_to_f_ratio', 


                    district_filter = district_filter

                )




            elif calc == 'YOY':


                yoy_calc(


                    df = df_, 


                    col = 'quarter', 


                    metric = 'd_to_f_ratio', 


                    district_filter = district_filter


                )



            elif calc == 'YTQ':



                ytm_ytq_calc(

                    df = df_, 

                    col = 'quarter', 

                    metric = 'd_to_f_ratio', 

                    district_filter = district_filter

                )



            else:

                pass



        else:


            pass



    
    
    


# Calling Calcs Function



st.write("<br><br>", unsafe_allow_html = True)




more_calcs()









# Footer Section



# Mention Data Source



st.write("<br><br><br><br>", unsafe_allow_html = True)




st.write(

        '''<footer class="css-164nlkn egzxvld1"><center><p>Data Source: <a href="https://data.telangana.gov.in/" target="_blank" class="css-1vbd788 egzxvld2">data.telangana.gov.in</a></p></center></footer>''', 


        unsafe_allow_html = True


        )










