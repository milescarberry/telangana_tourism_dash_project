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


import json


from streamlit_autorefresh import st_autorefresh


st.set_page_config(


    page_title="Telangana Tourism Dashboard",


    layout='wide'


)


pio.templates.default = 'ggplot2'


warnings.filterwarnings("ignore", category=DeprecationWarning)


warnings.filterwarnings("ignore", category=FutureWarning)


# Update dataframe every 6 hours


refresh_count = st_autorefresh(

	# interval = 6 * 60 * 60 * 1000,

	interval=5 * 60 * 1000,

	key="dataframerefresh"

	)


@st.cache_data
def get_dataset():

    query_list = ["python", "data_preprocessing.py"]

    call(query_list)

    df = pd.read_parquet("./cleaned_data/dom_df.parquet")

    return df


# Dashboard Title
st.write(

    "<h1 class='dashtitle'><center>Telangana Tourism Tracker: Domestic and Foreign Visitors (2016-2019)</center></h1>",


    unsafe_allow_html=True


)


st.write("<br>", unsafe_allow_html=True)


# Some More Text


# st.write(

#     "<h5 class='dashsubtitle'><center>Some more text comes here.</center></h5>",

#     unsafe_allow_html=True

# )


# st.write("<br>", unsafe_allow_html=True)


# Getting Data


dom_df = get_dataset()


# Sidebar Filters


with st.sidebar:

    # Filters Title

    st.write("<h1><center>Filters</center></h1><br>", unsafe_allow_html=True)

    if 'calcs' not in st.session_state:

        st.session_state.calcs = {


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

    # Year Filter

    years_list = ['All']

    years = list(dom_df['year'].unique())

    years.sort()

    years_list.extend(years)

    if 'year_len' not in st.session_state:

        st.session_state['year_len'] = 1

    if "calc_index" not in st.session_state:

        st.session_state.calc_index = 0

    def year_multiselect_callback_func():

        if len(st.session_state.new_year_len) == 0:

            st.session_state.new_year_len = [2019]

        if 'All' in st.session_state.new_year_len:

            st.session_state.new_year_len = years

        st.session_state.year_len = len(st.session_state.new_year_len)

        if st.session_state.year_len > 1:

            st.session_state.calc_index = 0

            st.session_state.calcs['Month'] = [
                'Percent Change From Previous Month']

            st.session_state.calcs['Quarter'] = [
                'Percent Change From Previous Quarter']

        else:

            if len(st.session_state.new_year_len) == 1 and 2016 in st.session_state.new_year_len:

                st.session_state.calcs['Month'] = [

                    'Percent Change From Previous Month',

                    # 'YOY',

                    'YTM'

                ]

                st.session_state.calcs['Quarter'] = [

                    "Percent Change From Previous Quarter",

                    # 'YOY',

                    'YTQ'

                ]

                st.session_state.calc_index = 0

            elif len(st.session_state.new_year_len) == 1 and st.session_state.calc_index == 1:

                # st.write("Yay!")

                n_year_len = []

                for y in st.session_state.new_year_len:

                    n_year_len.append(y - 1)

                    n_year_len.append(y)

                st.session_state.new_year_len = n_year_len

            else:

                st.session_state.calcs['Month'] = [

                    'Percent Change From Previous Month',

                    'YOY',

                    'YTM'

                ]

                st.session_state.calcs['Quarter'] = [

                    "Percent Change From Previous Quarter",

                    'YOY',

                    'YTQ'

                ]

    year_filter = st.multiselect(


        label="Select Year: ",


        options=years_list,


        default=2019,


        on_change=year_multiselect_callback_func,


        key='new_year_len'

    )

    if 'All' in year_filter:

        year_filter = years

    elif len(year_filter) == 0:

        year_filter = [2019]

    else:

        pass

    # if len(year_filter) > 1:

    # 	calcs['Month'] = ['Percent Change From Previous Month']

    # 	calcs['Quarter'] = ['Percent Change From Previous Quarter']

    st.write("<br>", unsafe_allow_html=True)

    # District Filter

    district_filter = ['All']

    unique_districts = list(dom_df.district.unique())

    unique_districts.sort()

    district_filter.extend(unique_districts)

    def district_filt_callback_func():

        if len(st.session_state.new_district_filt) == 0:

            st.session_state.new_district_filt = ['Hyderabad']

        # elif 'All' in st.session_state.new_district_filt:

        # 	st.session_state.new_district_filt = unique_districts

        else:

        	pass


    district_filt = st.multiselect(


        "Select Districts: ",

        district_filter,

        default=['All'],

        on_change=district_filt_callback_func,

        key='new_district_filt'

    )

    st.write("<br>", unsafe_allow_html=True)

    # Time Axis Filter

    if 'time_ax' not in st.session_state:

        st.session_state.time_ax = 'Month'

    time_axes = ['Month', 'Quarter']

    st.session_state.time_ax = st.selectbox(


        "Select Time Axis: ",

        time_axes,

        index=0

    )

    st.write("<br>", unsafe_allow_html=True)

    # Metric Filter

    metrics = [

        'Domestic Visitors',

        'Foreign Visitors',

        'Domestic and Foreign Visitors',

        # 'Domestic to Foreign Visitor Ratio'

    ]

    metric = st.selectbox(


        "Select Metric: ",

        metrics,

        index=0

    )

    st.write("<br>", unsafe_allow_html=True)

    # Calculation Filter

    def calc_callback_func():

        st.session_state.calc_index = st.session_state.calcs[

            st.session_state.time_ax

        ].index(st.session_state.new_calc)

        year_multiselect_callback_func()

    calc = st.selectbox(



        "Select Calculation: ",


        st.session_state.calcs[st.session_state.time_ax],


        index=st.session_state.calc_index,


        on_change=calc_callback_func,


        key='new_calc',


        help=f"\n{'YOY'.upper()}: {'Year on Year Percentage Change.'.title()} {'(Percentage change from previous years month/quarter with current years month/quarter)'.title()}\n\n{'YTM'.upper()}: {'Year to Month Percentage Change. (Percentage Change from First Month of the Year to the current month)'.title()}\n\n{'YTQ'.upper()}: {'Year to Quarter Percentage change. (Percentage change from first quarter of the year to the current quarter)'.title()}\n"


    )


# Update Districts


def update_districts(df):

    dists_filt = []

    if 'All' in st.session_state.new_district_filt:

        dists_filt = list(df.district.unique())

    else:

        dists_filt = st.session_state.new_district_filt

    return df[df.district.isin(dists_filt)]


# Apply Year Filter
dom_df = dom_df[dom_df['year'].isin(st.session_state.new_year_len)]


# Topside Metrics (KPIs)


# Creating 3 Columns


met_df = update_districts(df=dom_df)


metcol1, metcol2, metcol3 = st.columns(3)


total_domestic_visitors = met_df['domestic_visitors'].sum()


total_foreign_visitors = met_df['foreign_visitors'].sum()


overall_d_to_f = 0


try:

	overall_d_to_f = int(
	    round(total_domestic_visitors / total_foreign_visitors, 0))


except:

	overall_d_to_f = np.nan


else:

	pass


with metcol1:

    with st.container(border=True):

        st.metric(


            label="Domestic Visitors",


            # value = f"{total_domestic_visitors:,}"


            value=f"{total_domestic_visitors / 1000000:.1f}M"


        )


with metcol2:

    with st.container(border=True):

        st.metric(

            label="Foreign Visitors",

            # value = f"{total_foreign_visitors:,}",

            value=f"{total_foreign_visitors / 1000:.1f}K"

        )


with metcol3:

    with st.container(border=True):

        st.metric(

            label="Domestic to Foreign Visitor Ratio",

            value=overall_d_to_f

        )


st.write("<br>", unsafe_allow_html=True)


# Basic Stats Chart

def plot_stats(visitor_type=metric, district_filter=district_filt, time_axis=st.session_state.time_ax):

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

                as_index=False,

                dropna=False

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


                    x=fig_df['date'],


                    y=fig_df['domestic_visitors'],


                    # mode='lines+markers+text',


                    mode='lines+markers',


                    name='Domestic',


                    # text = fig_df['domestic_visitors'],


                    textposition='top center'


                ),



                secondary_y=False




            )

            fig.add_trace(


                go.Scatter(


                    x=fig_df['date'],


                    y=fig_df['foreign_visitors'],


                    # mode='lines+markers+text',


                    mode='lines+markers',


                    name='Foreign',


                    # text = fig_df['foreign_visitors'],


                    textposition='bottom center'


                ),



                secondary_y=True



            )

            fig.update_xaxes(title=' ')

        else:

            fig_df = fig_df.groupby(

                ['year_quarter'],

                as_index=False,

                dropna=False

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


                    x=fig_df['year_quarter'],


                    y=fig_df['domestic_visitors'],


                    # mode='lines+markers+text',


                    mode='lines+markers',


                    name='Domestic',


                    # text = fig_df['domestic_visitors'],


                    textposition='top center'


                ),



                secondary_y=False




            )

            fig.add_trace(

                go.Scatter(


                    x=fig_df['year_quarter'],


                    y=fig_df['foreign_visitors'],


                    # mode='lines+markers+text',


                    mode='lines+markers',


                    name='Foreign',


                    # text = fig_df['foreign_visitors'],


                    textposition='bottom center'



                ),


                secondary_y=True



            )

            fig.update_xaxes(title=' ', tickangle=-55)

        fig.update_yaxes(title="Domestic Visitors", secondary_y=False)

        fig.update_yaxes(title="Foreign Visitors", secondary_y=True)

        fig.update_yaxes(secondary_y=True, showgrid=False)

        hovertemp = "<br><br>".join([

            "<b>%{x}</b>",

            "<b>%{y:.2s} visitors</b><extra></extra>"


        ])

        fig.update_traces(hovertemplate=hovertemp)

        if len(district_filter) == 33:

            fig.update_layout(

                title=dict(

                    text=f"Domestic vs. Foreign Visitors for All Districts",

                    x=0.5,

                    xanchor='center',

                    yanchor='top',


                    font=dict(size=14.5)

                )

            )

        else:

            fig.update_layout(

                title=dict(


                    text=f"Domestic vs. Foreign Visitors By {st.session_state.time_ax.title()} For {district_filter[0] if len(district_filter) == 1 else 'The Selected Districts'}",


                    x=0.5,


                    xanchor='center',

                    yanchor='top',


                    font=dict(size=14.5)


                )

            )

        fig.update_layout(width=400, height=450)

        st.plotly_chart(fig, use_container_width=True)

    elif visitor_type == 'Domestic Visitors':

        if time_axis == 'Month':

            fig_df = fig_df.groupby(

                ['date'],

                as_index=False,

                dropna=False

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

                labels={'x': ' ', 'y': 'Domestic Visitors'},

                markers=True,

                title='',


                # text='domestic_visitors'


            )

            fig.update_xaxes(title=' ')

        else:

            fig_df = fig_df.groupby(

                ['year_quarter'],

                as_index=False,

                dropna=False

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

                labels={'x': ' ', 'y': 'Domestic Visitors'},

                markers=True,

                title='',

                # text='domestic_visitors'


            )

            fig.update_xaxes(title=' ', tickangle=-55)

        fig.update_yaxes(title="Domestic Visitors")

        hovertemp = "<br><br>".join([


            "<b>%{x}</b>",


            "<b>%{y:.2s} visitors</b><extra></extra>"


        ])

        fig.update_traces(hovertemplate=hovertemp,
                          texttemplate="<b>%{text:.2s}</b>", textposition='top center')

        if len(district_filter) == 33:

            fig.update_layout(

                title=dict(

                    text=f"Domestic Visitors for All Districts",


                    x=0.5,


                    xanchor='center',

                    yanchor='top',


                    font=dict(size=14.5)


                )

            )

        else:

            fig.update_layout(

                title=dict(

                    text=f"Domestic Visitors By {st.session_state.time_ax.title()} For {district_filter[0] if len(district_filter) == 1 else 'The Selected Districts'}",


                    x=0.5,


                    xanchor='center',

                    yanchor='top',

                    font=dict(size=14.5)


                )

            )

        fig.update_layout(height=450)

        st.plotly_chart(fig, use_container_width=True)

    elif visitor_type == 'Foreign Visitors':

        if time_axis == 'Month':

            fig_df = fig_df.groupby(

                ['date'],

                as_index=False,

                dropna=False

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

                labels={'x': ' ', 'y': 'Foreign Visitors'},

                markers=True,

                title='',

                # text='foreign_visitors'


            )

            fig.update_xaxes(title=' ')

        else:

            fig_df = fig_df.groupby(

                ['year_quarter'],

                as_index=False,

                dropna=False

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

                labels={'x': ' ', 'y': 'Foreign Visitors'},

                markers=True,

                title='',


                # text='foreign_visitors'


            )

            fig.update_xaxes(title=' ', tickangle=-55)

        fig.update_yaxes(title="Foreign Visitors")

        hovertemp = "<br><br>".join([


            "<b>%{x}</b>",


            "<b>%{y:.2s} visitors</b><extra></extra>"


        ])

        fig.update_traces(hovertemplate=hovertemp,
                          texttemplate="<b>%{text:.2s}</b>", textposition='top center')

        if len(district_filter) == 33:

            fig.update_layout(

                title=dict(

                    text=f"Foreign Visitors for All Districts",


                    x=0.5,


                    xanchor='center',

                    yanchor='top',

                    font=dict(size=14.5)


                )

            )

        else:

            fig.update_layout(


                title=dict(


                    text=f"Foreign Visitors By {st.session_state.time_ax.title()} For {district_filter[0] if len(district_filter) == 1 else 'The Selected Districts'}",


                    x=0.5,


                    xanchor='center',

                    yanchor='top',


                    font=dict(size=14.5)


                )

            )

        fig.update_layout(height=450)

        st.plotly_chart(fig, use_container_width=True)

    else:

        if time_axis == 'Month':

            fig_df = fig_df.groupby(

                ['date'],

                as_index=False,

                dropna=False

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

                labels={'x': ' ', 'y': 'd_to_f_ratio'},

                markers=True,

                title='',


                # text='d_to_f_ratio'


            )

            fig.update_xaxes(title=' ')

        else:

            fig_df = fig_df.groupby(

                ['year_quarter'],

                as_index=False,

                dropna=False

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

                labels={'x': ' ', 'y': 'd_to_f_ratio'},

                markers=True,

                title='',

                # text='d_to_f_ratio'


            )

            fig.update_xaxes(title=' ', tickangle=-55)

        fig.update_yaxes(title="D to F Ratio")

        hovertemp = "<br><br>".join([


            "<b>%{x}</b>",


            "<b>D to F Ratio: %{y:.2s}</b><extra></extra>"


        ])

        fig.update_traces(hovertemplate=hovertemp,
                          texttemplate="<b>%{text:.2s}</b>", textposition='top center')

        if len(district_filter) == 33:

            fig.update_layout(

                title=dict(

                    text=f"Domestic to Foreign Visitor Ratio for All Districts",


                    x=0.5,

                    xanchor='center',

                    yanchor='top',


                    font=dict(size=14.5)


                )

            )

        else:

            fig.update_layout(

                title=dict(

                    text=f"Domestic to Foreign Visitor Ratio By {st.session_state.time_ax.title()} For {district_filter[0] if len(district_filter) == 1 else 'The Selected Districts'}",


                    x=0.5,


                    xanchor='center',

                    yanchor='top',


                    font=dict(size=14.5)


                )

            )

        fig.update_layout(height=450)

        st.plotly_chart(fig, use_container_width=True)


# plot_stats()


# Calc Chart


# Calc Funcs


# Percent Difference Func


def percent_change_from_previous(df, col='month', metric='domestic_visitors', district_filter=[]):

    time_val = ''

    if col == 'month':

        time_val = 'date'

    else:

        time_val = 'year_quarter'

        df['date'] = df.year_quarter.apply(
            lambda x: dom_df[dom_df.year_quarter == x]['date'].min())

    if 'list' not in str(type(metric)).lower():

        df[f'prev_{metric}'] = df[metric].shift(1)

        df[f'{metric}_percent_change'] = (

            (df[metric] - df[f'prev_{metric}']) / df[f'prev_{metric}']

        ) * 100

        fig = exp.line(

            df,

            x=time_val,

            y=f'{metric}_percent_change',

            labels={'x': ' ', 'y': f'{metric}_percent_change'},

            markers=True,

            title='',


            # text=f'{metric}_percent_change',



            custom_data=[df[metric], df[f"prev_{metric}"]]


        )

        if time_val == 'year_quarter':

            fig.update_xaxes(title=' ', tickangle=-55)

        else:

            fig.update_xaxes(title=' ')

        # fig.update_yaxes(title = f'{metric}_percent_change'.replace("_", ' ').title())

        fig.update_yaxes(title="% Change")

        hovertemp = "<br><br>".join([


            "<b>%{x}</b>",


            f"<b></b>" +
            "<b>%{customdata[0]:.2s} visitors</b>",


            f"<b>Previous {st.session_state.time_ax}: </b>" +
            "<b>%{customdata[1]:.2s} visitors</b>",


            "<b>%{y:.2s}%</b>" +
            f" <b>vs. Previous {st.session_state.time_ax}</b><extra></extra>"


        ])

        fig.update_traces(hovertemplate=hovertemp,
                          texttemplate="<b>%{text:.1f}%</b>", textposition='top center')

        if len(district_filter) == 33:

            fig.update_layout(

                title=dict(

                    text=f"{metric.replace('_', ' ').title()} Percent Change From Previous {st.session_state.time_ax.title()} For All Districts",


                    x=0.5,

                    xanchor='center',

                    yanchor='top',


                    font=dict(size=13.5)

                )

            )

        else:

            fig.update_layout(

                title=dict(

                    text=f"{metric.replace('_', ' ').title()} Percent Change From Previous {st.session_state.time_ax.title()} By {st.session_state.time_ax.title()} For {district_filter[0] if len(district_filter) == 1 else 'The Selected Districts'}",


                    x=0.5,

                    xanchor='center',

                    yanchor='top',


                    font=dict(size=13.5)

                )

            )

        fig.update_layout(height=450)

        st.plotly_chart(fig, use_container_width=True)

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


                # mode='lines+markers+text',


                mode='lines+markers',


                # text = df[f'{metric[0]}_percent_change'],


                # textposition = 'top center',


                name='Domestic',


                customdata=df[

                    [i for i in df.columns.values if 'foreign' not in i]

                ]


            ),



            secondary_y=False




        )

        fig.add_trace(


            go.Scatter(


                x=df[time_val],


                y=df[f'{metric[1]}_percent_change'],


                # mode='lines+markers+text',

                mode='lines+markers',


                # text = df[f'{metric[1]}_percent_change'],


                textposition='bottom center',


                name='Foreign',


                customdata=df[

                    [i for i in df.columns.values if 'domestic' not in i]

                ]




            ),



            secondary_y=True



        )

        fig.update_yaxes(title="Domestic Visitors % Change",
                         secondary_y=False, showgrid=True)

        fig.update_yaxes(title="Foreign Visitors % Change",
                         secondary_y=True, showgrid=False)

        if time_val == 'year_quarter':

            fig.update_xaxes(title=' ', tickangle=-55)

        else:

            fig.update_xaxes(title=' ')

        if 'month' in st.session_state.time_ax.lower():

            hovertemp = "<br><br>".join([


                "<b>%{x}</b>",


                f"<b></b>" +
                "<b>%{customdata[1]:.2s} visitors</b>",


                f"<b>Previous {st.session_state.time_ax}: </b>" +
                "<b>%{customdata[2]:.2s} visitors</b>",


                "<b>%{y:.2s}%</b>" +
                f" <b>vs. Previous {st.session_state.time_ax}</b><extra></extra>"


            ])

        else:

            hovertemp = "<br><br>".join([


                "<b>%{x}</b>",


                f"<b></b>" +
                "<b>%{customdata[1]:.2s} visitors</b>",


                f"<b>Previous {st.session_state.time_ax}: </b>" +
                "<b>%{customdata[3]:.2s} visitors</b>",


                "<b>%{y:.2s}%</b>" +
                f" <b>vs. Previous {st.session_state.time_ax}</b><extra></extra>"


            ])

        fig.update_traces(hovertemplate=hovertemp)

        if len(district_filter) == 33:

            fig.update_layout(

                title=dict(

                    text=f"{metric[0].replace('_', ' ').title()} vs. {metric[1].replace('_', ' ').title()} Percent Change From Previous {st.session_state.time_ax.title()} For All Districts",


                    x=0.5,

                    xanchor='center',

                    yanchor='top',


                    font=dict(size=13.5)

                )

            )

        else:

            fig.update_layout(

                title=dict(

                    text=f"{metric[0].replace('_', ' ').title()} vs. {metric[1].replace('_', ' ').title()} Percent Change From Previous {st.session_state.time_ax.title()} For {district_filter[0] if len(district_filter) == 1 else 'The Selected Districts'}",

                    x=0.5,

                    xanchor='center',

                    yanchor='top',


                    font=dict(size=13.5)

                )

            )

        fig.update_layout(height=450)

        st.plotly_chart(fig, use_container_width=True)


# YOY Func


def yoy_calc(df, col='month', metric='domestic_visitors', district_filter=[]):

    time_val = ''

    if col == 'month':

        time_val = 'date'

    else:

        time_val = 'year_quarter'

        df['date'] = df.year_quarter.apply(
            lambda x: dom_df[dom_df.year_quarter == x]['date'].min())

    if 'list' not in str(type(metric)).lower():

        df['prev_year_date'] = df['date'].apply(
            lambda x: x - dt.timedelta(365))

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


            (df[metric] - df[f'prev_year_{metric}']
             ) / df[f'prev_year_{metric}']


        ) * 100

        fig = exp.line(

            df,

            x=time_val,

            y=f'{metric}_prev_year_percent_change',

            labels={'x': ' ', 'y': f'{metric}_prev_year_percent_change'},

            markers=True,

            title='',


            # text=f'{metric}_prev_year_percent_change',



            custom_data=[df[metric], df[f"prev_year_{metric}"]]


        )

        fig.update_xaxes(title=' ', tickangle=-55)

        # fig.update_yaxes(title = f'{metric}_prev_year_percent_change'.replace("_", ' ').title())

        fig.update_yaxes(title="% Change")

        if time_val == 'year_quarter':

            fig.update_xaxes(title=' ', tickangle=-55)

        else:

            fig.update_xaxes(title=' ')

        hovertemp = "<br><br>".join([


            "<b>%{x}</b>",


            f"<b></b>" +
            "<b>%{customdata[0]:.2s} visitors</b>",


            f"<b>Previous Year: </b>" +
            "<b>%{customdata[1]:.2s} visitors</b>",


            "<b>%{y:.2f} %</b>" + " <b>vs. Previous Year</b><extra></extra>"


        ])

        fig.update_traces(hovertemplate=hovertemp,
                          texttemplate="<b>%{text:.1f}%</b>", textposition='top center')

        if len(district_filter) == 33:

            fig.update_layout(

                title=dict(

                    text=f"{metric.replace('_', ' ').title()} YOY Percent Change For All Districts",


                    x=0.5,

                    xanchor='center',

                    yanchor='top',


                    font=dict(size=13.5)

                )

            )

        else:

            fig.update_layout(

                title=dict(

                    text=f"{metric.replace('_', ' ').title()} YOY Percent Change By {st.session_state.time_ax.title()} For {district_filter[0] if len(district_filter) == 1 else 'The Selected Districts'}",

                    x=0.5,

                    xanchor='center',

                    yanchor='top',


                    font=dict(size=13.5)

                )

            )

        fig.update_layout(height=450)

        st.plotly_chart(fig, use_container_width=True)

    else:

        df['prev_year_date'] = df['date'].apply(
            lambda x: x - dt.timedelta(365))

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


                x=df[time_val],


                y=df[f'{metric[0]}_prev_year_percent_change'],


                # mode='lines+markers+text',


                mode='lines+markers',


                # text = df[f'{metric[0]}_prev_year_percent_change'],


                # textposition = 'top center',


                name='Domestic',


                customdata=df[

                    [i for i in df.columns.values if 'foreign' not in i]

                ]


            ),



            secondary_y=False




        )

        fig.add_trace(


            go.Scatter(


                x=df[time_val],


                y=df[f'{metric[1]}_prev_year_percent_change'],


                # mode='lines+markers+text',


                mode='lines+markers',


                # text = df[f'{metric[1]}_prev_year_percent_change'],


                # textposition = 'bottom center',


                name='Foreign',


                customdata=df[

                    [i for i in df.columns.values if 'domestic' not in i]

                ]




            ),



            secondary_y=True



        )

        fig.update_yaxes(
            title=f"{metric[0].replace('_', ' ').title()} % Change", secondary_y=False, showgrid=True)

        fig.update_yaxes(
            title=f"{metric[1].replace('_', ' ').title()} % Change", secondary_y=True, showgrid=False)

        if time_val == 'year_quarter':

            fig.update_xaxes(title=' ', tickangle=-55)

        else:

            fig.update_xaxes(title=' ')

        if 'month' not in st.session_state.time_ax.lower():

            hovertemp = "<br><br>".join([


                "<b>%{x}</b>",

                "<b>Previous Year's Figure: </b>" +
                "<b>%{customdata[4]:.2s} visitors</b>",

                "<b>Current Year's Figure: </b>" +
                "<b>%{customdata[1]:.2s} visitors</b>",

                "<b>%{y:.2f} % vs. Previous Year</b><extra></extra>"


            ])

        else:

            hovertemp = "<br><br>".join([

                "<b>%{x}</b>",

                "<b>Previous Year's Figure: </b>" +
                "<b>%{customdata[3]:.2s} visitors</b>",

                "<b>Current Year's Figure: </b>" +
                "<b>%{customdata[1]:.2s} visitors</b>",

                "<b>%{y:.2f} % vs. Previous Year</b><extra></extra>"


            ])

        fig.update_traces(hovertemplate=hovertemp)

        if len(district_filter) == 33:

            fig.update_layout(

                title=dict(

                    text=f"{metric[0].replace('_', ' ').title()} vs. {metric[1].replace('_', ' ').title()} YOY Percent Change For All Districts",


                    x=0.5,

                    xanchor='center',

                    yanchor='top',


                    font=dict(size=13.5)

                )

            )

        else:

            fig.update_layout(

                title=dict(

                    text=f"{metric[0].replace('_', ' ').title()} vs. {metric[1].replace('_', ' ').title()} YOY Percent Change By {st.session_state.time_ax.title()} For {district_filter[0] if len(district_filter) == 1 else 'The Selected Districts'}",


                    x=0.5,

                    xanchor='center',

                    yanchor='top',


                    font=dict(size=13.5)

                )

            )

        fig.update_layout(height=450)

        st.plotly_chart(fig, use_container_width=True)


# YTM / YTQ Func


def ytm_ytq_calc(df, col='month', metric='domestic_visitors', district_filter=[]):

    time_val = ''

    if col == 'month':

        time_val = 'date'

    else:

        time_val = 'year_quarter'

        df['date'] = df.year_quarter.apply(
            lambda x: dom_df[dom_df.year_quarter == x]['date'].min())

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


            (df[metric] -
             df[f'first_date_year_{metric}']) / df[f'first_date_year_{metric}']


        ) * 100

        fig = exp.line(

            df,

            x=time_val,

            y=f'{metric}_first_date_year_percent_change',

            labels={'x': ' ', 'y': f'{metric}_first_date_year_percent_change'},

            markers=True,

            title='',


            # text=f'{metric}_first_date_year_percent_change',



            custom_data=[df[metric], df[f'first_date_year_{metric}']]


        )

        # fig.update_yaxes(title = f'{metric}_first_date_year_percent_change'.replace('_', ' ').title())

        fig.update_yaxes(title="% Change")

        if time_val == 'year_quarter':

            fig.update_xaxes(title=' ', tickangle=-55)

        else:

            fig.update_xaxes(title=' ')

        if time_val == 'date':

            hovertemp = "<br><br>".join([


                "<b>%{x}</b>",


                f"<b>Total {' '.join(metric.split('_')).title()} In First {st.session_state.time_ax}: </b>" +
                "<b>%{customdata[1]:.2s}</b>",


                f"<b>Total {' '.join(metric.split('_')).title()} In Current Month: </b>" +
                "<b>%{customdata[0]:.2s}</b>",


                " <b>YTM: </b>" + "<b>%{y:.2f} %</b><extra></extra>"


            ])

        else:

            hovertemp = "<br><br>".join([


                "<b>%{x}</b>",


                f"<b>Total {' '.join(metric.split('_')).title()} In First {st.session_state.time_ax}: </b>" +
                "<b>%{customdata[1]:.2s}</b>",


                f"<b>Total {' '.join(metric.split('_')).title()} In Current Quarter: </b>" +
                "<b>%{customdata[0]:.2s}</b>",


                " <b>YTQ: </b>" + "<b>%{y:.2f} %</b><extra></extra>"


            ])

        fig.update_traces(hovertemplate=hovertemp,
                          texttemplate="<b>%{text:.1f}%</b>", textposition='top center')

        if time_val == 'date':

            if len(district_filter) == 33:

                fig.update_layout(

                    title=dict(

                        text=f"{metric.replace('_', ' ').title()} YTM Percent Change For All Districts",


                        x=0.5,

                        xanchor='center',

                        yanchor='top',


                        font=dict(size=13.5)

                    )

                )

            else:

                fig.update_layout(

                    title=dict(

                        text=f"{metric.replace('_', ' ').title()} YTM Percent Change For {district_filter[0] if len(district_filter) == 1 else 'The Selected Districts'}",


                        x=0.5,

                        xanchor='center',

                        yanchor='top',


                        font=dict(size=13.5)




                    )

                )

        else:

            if len(district_filter) == 33:

                fig.update_layout(

                    title=dict(

                        text=f"{metric.replace('_', ' ').title()} YTQ Percent Change For All Districts",


                        x=0.5,

                        xanchor='center',

                        yanchor='top',


                        font=dict(size=13.5)

                    )

                )

            else:

                fig.update_layout(

                    title=dict(

                        text=f"{metric.replace('_', ' ').title()} YTQ Percent Change For {district_filter[0] if len(district_filter) == 1 else 'The Selected Districts'}",


                        x=0.5,

                        xanchor='center',

                        yanchor='top',


                        font=dict(size=13.5)

                    )

                )

        fig.update_layout(height=450)

        st.plotly_chart(fig, use_container_width=True)

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


                (df[m] - df[f'first_date_year_{m}']
                 ) / df[f'first_date_year_{m}']


            ) * 100

        # Dual Axis Chart

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(


            go.Scatter(


                x=df[time_val],


                y=df[f'{metric[0]}_first_date_year_percent_change'],


                # mode='lines+markers+text',


                mode='lines+markers',


                # text = df[f'{metric[0]}_first_date_year_percent_change'],


                # text_position = 'top center',


                name='Domestic',


                customdata=df[[
                    i for i in df.columns.values if 'foreign' not in i]]


            ),



            secondary_y=False




        )

        fig.add_trace(


            go.Scatter(


                x=df[time_val],


                y=df[f'{metric[1]}_first_date_year_percent_change'],


                # mode='lines+markers+text',


                mode='lines+markers',


                # text = df[f'{metric[1]}_first_date_year_percent_change'],


                # text_position = 'bottom center',


                name='Foreign',


                customdata=df[[
                    i for i in df.columns.values if 'domestic' not in i]]




            ),



            secondary_y=True



        )

        if time_val == 'year_quarter':

            fig.update_xaxes(title=' ', tickangle=-55)

        else:

            fig.update_xaxes(title=' ')

        if time_val == 'date':

            fig.update_yaxes(
                title=f"{metric[0].replace('_', ' ').title()} % Change", secondary_y=False, showgrid=True)

            fig.update_yaxes(
                title=f"{metric[1].replace('_', ' ').title()} % Change", secondary_y=True, showgrid=False)

            hovertemp = "<br><br>".join([

                "<b>%{x}</b>",

                f"<b>First {st.session_state.time_ax}: </b>" +
                "<b>%{customdata[3]:.2s} visitors</b>",

                f"<b>This {st.session_state.time_ax}: </b>" +
                "<b>%{customdata[1]:.2s} visitors</b>",

                "<b>YTM: </b>" + "<b>%{y:.2f} %</b><extra></extra>"


            ])

        else:

            fig.update_yaxes(
                title=f"{metric[0].replace('_', ' ').title()} % Change", secondary_y=False, showgrid=True)

            fig.update_yaxes(
                title=f"{metric[1].replace('_', ' ').title()} % Change", secondary_y=True, showgrid=False)

            hovertemp = "<br><br>".join([

                "<b>%{x}</b>",

                f"<b>First {st.session_state.time_ax}: </b>" +
                "<b>%{customdata[4]:.2s} visitors</b>",

                f"<b>This {st.session_state.time_ax}: </b>" +
                "<b>%{customdata[1]:.2s} visitors</b>",

                "<b>YTQ: </b>" + "<b>%{y:.2f} %</b><extra></extra>"


            ])


        fig.update_traces(hovertemplate=hovertemp)


        if time_val == 'date':

            if len(district_filter) == 33:

                fig.update_layout(

                    title=dict(

                        text=f"{metric[0].replace('_', ' ').title()} vs. {metric[1].replace('_', ' ').title()} YTM Percent Change For All Districts",


                        x=0.5,

                        xanchor='center',

                        yanchor='top',


                        font=dict(size=13.5)

                    )

                )

            else:

                fig.update_layout(

                    title=dict(

                        text=f"{metric[0].replace('_', ' ').title()} vs. {metric[1].replace('_', ' ').title()} YTM Percent Change For {district_filter[0] if len(district_filter) == 1 else 'The Selected Districts'}",


                        x=0.5,

                        xanchor='center',

                        yanchor='top',


                        font=dict(size=13.5)

                    )

                )

        else:

            if len(district_filter) == 33:

                fig.update_layout(

                    title=dict(

                        text=f"{metric[0].replace('_', ' ').title()} vs. {metric[1].replace('_', ' ').title()} YTQ Percent Change For All Districts",


                        x=0.5,

                        xanchor='center',

                        yanchor='top',


                        font=dict(size=13.5)

                    )

                )

            else:

                fig.update_layout(

                    title=dict(

                        text=f"{metric[0].replace('_', ' ').title()} vs. {metric[1].replace('_', ' ').title()} YTQ Percent Change For {district_filter[0] if len(district_filter) == 1 else 'The Selected Districts'}",


                        x=0.5,

                        xanchor='center',

                        yanchor='top',


                        font=dict(size=13.5)

                    )

                )

        fig.update_layout(height=450)

        st.plotly_chart(fig, use_container_width=True)



# The Calcs Chart


def more_calcs(


    metric=metric,


    time_axis=st.session_state.time_ax,


    calc=calc,


    district_filter=district_filt

):

    if 'All' in district_filter:

        district_filter = unique_districts

    else:

        district_filter = district_filter

    df_ = dom_df

    df_ = df_[df_.district.isin(district_filter)]

    if time_axis == 'Month':

        if metric == 'Domestic Visitors':

            df_ = df_.groupby(

                ['date'],

                dropna=False,

                as_index=False

            ).agg(


                {"domestic_visitors": pd.Series.sum}

            )

            if calc == 'Percent Change From Previous Month':

                percent_change_from_previous(

                    df=df_,


                    col='month',


                    metric='domestic_visitors',


                    district_filter=district_filter


                )

            elif calc == 'YOY':

                yoy_calc(

                    df=df_,

                    col='month',

                    metric='domestic_visitors',

                    district_filter=district_filter

                )

            elif calc == 'YTM':

                ytm_ytq_calc(

                    df=df_,

                    col='month',

                    metric='domestic_visitors',

                    district_filter=district_filter

                )

            else:

                pass

        elif metric == 'Foreign Visitors':

            df_ = df_.groupby(

                ['date'],

                dropna=False,

                as_index=False

            ).agg(


                {"foreign_visitors": pd.Series.sum}

            )

            if calc == 'Percent Change From Previous Month':

                percent_change_from_previous(

                    df=df_,


                    col='month',


                    metric='foreign_visitors',


                    district_filter=district_filter


                )

            elif calc == 'YOY':

                yoy_calc(

                    df=df_,

                    col='month',

                    metric='foreign_visitors',

                    district_filter=district_filter


                )

            elif calc == 'YTM':

                ytm_ytq_calc(

                    df=df_,

                    col='month',

                    metric='foreign_visitors',

                    district_filter=district_filter

                )

            else:

                pass

        elif metric == 'Domestic and Foreign Visitors':

            df_ = df_.groupby(

                ['date'],

                dropna=False,

                as_index=False

            ).agg(


                {

                    "domestic_visitors": pd.Series.sum,


                    "foreign_visitors": pd.Series.sum

                }

            )

            if calc == 'Percent Change From Previous Month':

                percent_change_from_previous(

                    df=df_,


                    col='month',


                    metric=['domestic_visitors', 'foreign_visitors'],


                    district_filter=district_filter


                )

            elif calc == 'YOY':

                yoy_calc(

                    df=df_,

                    col='month',

                    metric=['domestic_visitors', 'foreign_visitors'],

                    district_filter=district_filter

                )

            elif calc == 'YTM':

                ytm_ytq_calc(

                    df=df_,

                    col='month',

                    metric=['domestic_visitors', 'foreign_visitors'],

                    district_filter=district_filter

                )

            else:

                pass

        elif metric == 'Domestic to Foreign Visitor Ratio':

            df_ = df_.groupby(

                ['date'],

                dropna=False,

                as_index=False

            ).agg(

                {"d_to_f_ratio": pd.Series.mean}

            )

            if calc == 'Percent Change From Previous Month':

                percent_change_from_previous(

                    df=df_,


                    col='month',


                    metric='d_to_f_ratio',


                    district_filter=district_filter


                )

            elif calc == 'YOY':

                yoy_calc(

                    df=df_,

                    col='month',

                    metric='d_to_f_ratio',

                    district_filter=district_filter

                )

            elif calc == 'YTM':

                ytm_ytq_calc(

                    df=df_,

                    col='month',

                    metric='d_to_f_ratio',

                    district_filter=district_filter

                )

            else:

                pass

        else:

            pass

    else:

        if metric == 'Domestic Visitors':

            df_ = df_.groupby(

                ['year_quarter'],

                dropna=False,

                as_index=False

            ).agg(


                {"domestic_visitors": pd.Series.sum}

            )

            if calc == 'Percent Change From Previous Quarter':

                percent_change_from_previous(

                    df=df_,


                    col='quarter',


                    metric='domestic_visitors',


                    district_filter=district_filter


                )

            elif calc == 'YOY':

                yoy_calc(

                    df=df_,

                    col='quarter',

                    metric='domestic_visitors',

                    district_filter=district_filter


                )

            elif calc == 'YTQ':

                ytm_ytq_calc(

                    df=df_,

                    col='quarter',

                    metric='domestic_visitors',

                    district_filter=district_filter

                )

            else:

                pass

        elif metric == 'Foreign Visitors':

            df_ = df_.groupby(

                ['year_quarter'],

                dropna=False,

                as_index=False

            ).agg(


                {"foreign_visitors": pd.Series.sum}

            )

            if calc == 'Percent Change From Previous Quarter':

                percent_change_from_previous(

                    df=df_,


                    col='quarter',


                    metric='foreign_visitors',


                    district_filter=district_filter


                )

            elif calc == 'YOY':

                yoy_calc(

                    df=df_,

                    col='quarter',

                    metric='foreign_visitors',

                    district_filter=district_filter


                )

            elif calc == 'YTQ':

                ytm_ytq_calc(

                    df=df_,

                    col='quarter',

                    metric='foreign_visitors',

                    district_filter=district_filter

                )

            else:

                pass

        elif metric == 'Domestic and Foreign Visitors':

            df_ = df_.groupby(

                ['year_quarter'],

                dropna=False,

                as_index=False

            ).agg(


                {

                    "domestic_visitors": pd.Series.sum,


                    "foreign_visitors": pd.Series.sum

                }

            )

            if calc == 'Percent Change From Previous Quarter':

                percent_change_from_previous(

                    df=df_,


                    col='quarter',


                    metric=['domestic_visitors', 'foreign_visitors'],


                    district_filter=district_filter


                )

            elif calc == 'YOY':

                yoy_calc(

                    df=df_,

                    col='quarter',

                    metric=['domestic_visitors', 'foreign_visitors'],

                    district_filter=district_filter


                )

            elif calc == 'YTQ':

                ytm_ytq_calc(

                    df=df_,

                    col='quarter',

                    metric=['domestic_visitors', 'foreign_visitors'],

                    district_filter=district_filter

                )

            else:

                pass

        elif metric == 'Domestic to Foreign Visitor Ratio':

            df_ = df_.groupby(

                ['year_quarter'],

                dropna=False,

                as_index=False

            ).agg(

                {"d_to_f_ratio": pd.Series.mean}

            )

            if calc == 'Percent Change From Previous Quarter':

                percent_change_from_previous(

                    df=df_,


                    col='quarter',


                    metric='d_to_f_ratio',


                    district_filter=district_filter

                )

            elif calc == 'YOY':

                yoy_calc(


                    df=df_,


                    col='quarter',


                    metric='d_to_f_ratio',


                    district_filter=district_filter


                )

            elif calc == 'YTQ':

                ytm_ytq_calc(

                    df=df_,

                    col='quarter',

                    metric='d_to_f_ratio',

                    district_filter=district_filter

                )

            else:

                pass

        else:

            pass


# Calling Calcs Function


# st.write("<br><br>", unsafe_allow_html=True)


# more_calcs()


# Telangana Districts Visitors Choropleth Map Section


# District Choropleth Map Function


def district_choropleth(df, geojson):


    df = update_districts(df=df)


    exp.set_mapbox_access_token(st.secrets["mapbox_access_token"])


    dom_grp_df = pd.DataFrame()


    # df = df[df.district.isin(st.session_state.new_district_filt)]


    if metric != 'Domestic and Foreign Visitors':

        dom_grp_df = df.groupby(


            ['district'],

            as_index=False,

            dropna=False

        ).agg(


            {f"{'_'.join(metric.lower().split(' '))}": pd.Series.sum}


        )



        fig = exp.choropleth_mapbox(


            dom_grp_df,


            geojson=geojson,


            color=f"{'_'.join(metric.lower().split(' '))}",


            locations='district',


            featureidkey='properties.DISTRICT_N',


            color_continuous_scale=exp.colors.sequential.Oranges,


            center=dict(lat=18.107054923278803, lon=79.2766835839577),


            mapbox_style="carto-darkmatter",


            zoom=6.0,



            custom_data=[


                dom_grp_df.district,


                dom_grp_df[f"{'_'.join(metric.lower().split(' '))}"]


            ]



        )


        hovertemp = "<br><br>".join(


            [

                    "<b>%{customdata[0]}</b>",


                    f"<b>{metric}: </b>" +
                "<b>%{customdata[1]:.2s}</b><extra></extra>"


            ]


        )

        fig.update_traces(hovertemplate=hovertemp)




    else:


        dom_grp_df = df.groupby(


            ['district'],


            as_index=False,


            dropna=False


        ).agg(

            {

                "domestic_visitors": pd.Series.sum,


                "foreign_visitors": pd.Series.sum

            }


        )

        dom_grp_df['total_visitors'] = dom_grp_df.domestic_visitors + \
            dom_grp_df.foreign_visitors



        fig = exp.choropleth_mapbox(


            dom_grp_df,


            geojson=geojson,


            color="total_visitors",


            locations='district',


            featureidkey='properties.DISTRICT_N',


            color_continuous_scale=exp.colors.sequential.Oranges,


            center=dict(lat=18.107054923278803, lon=79.2766835839577),


            mapbox_style="carto-darkmatter",


            zoom=6.0,



            custom_data=[


                dom_grp_df.district,


                dom_grp_df["total_visitors"],


                dom_grp_df['domestic_visitors'],


                dom_grp_df['foreign_visitors']


            ]



        )

        hovertemp = "<br><br>".join(


            [

                "<b>%{customdata[0]}</b>",


                "<b>Domestic Visitors: %{customdata[2]:.2s}</b>",


                    "<b>Foreign Visitors: %{customdata[3]:.2s}</b>",


                    "<b>Total Visitors: %{customdata[1]:.2s}</b><extra></extra>"


            ]


        )

        fig.update_traces(hovertemplate=hovertemp)







    fig.update_layout(



        margin=dict(r=0, t=0, l=0, b=0)


    )

    fig.update_layout(

        coloraxis_colorbar=dict(title=' '),

        coloraxis_showscale=False

    )


    fig.update_layout(height = 509, width=400)


    st.plotly_chart(fig)



# Get District GeoJSON File


# GeoJSON Func


@st.cache_data
def get_districts_geojson():

    with open("./input_files/shape_files/TS_District_Boundary_33_FINAL.geojson", 'r') as gfile:

        gjson = gfile.read()

    gjson = json.loads(gjson)

    gjson['features'][16]['properties']['DISTRICT_N'] = 'Medchal Malkajgiri'

    gjson['features'][5]['properties']['DISTRICT_N'] = 'Jangaon'

    return gjson


gjson = get_districts_geojson()


# Display District Choropleth Map


# st.write("<br><br>", unsafe_allow_html = True)


# district_choropleth(df=dom_df, geojson=gjson)


# Dash Layout


# Create Two Columns


dcol1, dcol2 = st.columns([1.3, 2])


with dcol1:

    with st.container(border=True):

        district_choropleth(

        	df=dom_df, 

        	geojson=gjson

        	)


with dcol2:

    with st.container(border=True):

        # Creating 2 Tabs

        tab1, tab2 = st.tabs(["🡸", "🡺"])

        with tab1:

            plot_stats()

        with tab2:

            more_calcs()


# Styling


def styling_func():

    css = '''


		div[class^='st-emotion-cache-16txtl3'] { 


		 padding-top: 1rem; 


		}


		div[class^='block-container'] { 

		  padding-top: 1rem; 

		}


		[data-testid="stMetric"] {
		    width: fit-content;
		    margin: auto;
		}

		[data-testid="stMetric"] > div {
		    width: fit-content;
		    margin: auto;
		}

		[data-testid="stMetric"] label {
		    width: fit-content;
		    margin: auto;
		}


		[data-testid="stMarkdownContainer"] > p {

          font-weight: bold;

        }


        [data-testid="stMetricValue"] {


          font-weight: bold;


        }


        [class^='dashtitle'] {

          font-size: 34px;

        }


        [class^='dashsubtitle'] {

          font-size: 17px;

        }
          
        


	'''

    st.write(

        f"<style>{css}</style>",

        unsafe_allow_html=True)


styling_func()



# Footer Section


# Mention Data Source


st.write("<br><br><br><br>", unsafe_allow_html=True)


st.write(

    '''<footer class="css-164nlkn egzxvld1"><center><p>Data Source: <a href="https://data.telangana.gov.in/" target="_blank" class="css-1vbd788 egzxvld2">data.telangana.gov.in</a></p></center></footer>''',


    unsafe_allow_html=True


)
