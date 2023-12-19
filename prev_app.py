import numpy as np 

import pandas as pd 


from plotly import express as exp, graph_objects as go, io as pio 


pio.templates.default = 'ggplot2'


import matplotlib.pyplot as plt 


import seaborn as sns


import datetime as dt 


import streamlit as st 


import os



import pickle


from pickle import dump, load


import json


import time


import folium


from streamlit_folium import st_folium, folium_static





st.set_page_config(


	page_title = "TSSPDCL EV Charger Consumption", 


	layout = 'wide'


	)






@st.cache_data
def get_datasets():


	# Telangana Districts GeoJson



	with open("./TS_DISTRICT_BOUNDARIES_33.geojson", 'rb') as telangana_dists:


		ta_districts = json.load(telangana_dists)





	with open("./ev_grp_3.pkl", 'rb') as ev_file:


		data = pickle.load(ev_file)



	data = data.drop_duplicates()


	data = data.reset_index(drop = True)


	data.load = data.load.apply(lambda x: round(x, 0))


	data['year_month'] = data.apply(lambda x: dt.datetime.strptime(str(x[0]) + "-" + str(x[1]), "%Y-%m"), axis = 1)


	data['financial_year'] = data.apply(lambda x: x[0] - 1 if x[1] >= 1 and x[1] <= 3 else x[0], axis = 1)


	data['year_quarter'] = data.apply(lambda x: x[2] + '-'+ str(x[-1]), axis = 1)


	quarts = []


	for quart in data.quarter.values:


		if quart == 'Q1':


			quarts.append('06')


		elif quart == 'Q2':


			quarts.append('09')


		elif quart == 'Q3':


			quarts.append('12')


		else:


			quarts.append('03')


	data['quart'] = quarts


	data.quart = data.apply(lambda x: dt.datetime.strptime(x[-1] + '-' + str(x[0]), "%m-%Y"), axis = 1)




	return data, ta_districts




data, ta_districts = get_datasets()


# Dashboard Title


st.write(

	'''<h1><center>Tracking Energy Consumption from TSSPDCL EV Charging Stations</center></h1>''',

	unsafe_allow_html = True

	)


st.write("<br>", unsafe_allow_html = True)



# Brief Intro


st.write(

	'''<h5><center>The Telangana Southern Power Distribution Company Limited (TSSPDCL) is a public undertaking of the Government of Telangana, India that distributes electricity and facilitates access to electric vehicle charging (EV) stations in the southern districts of the State.<br><br>This dashboard attempts to track the energy cosumption from EV charging stations of various loads from June 2021 to July 2023.</center></h5>''', 


	unsafe_allow_html = True


	)



st.write("<br><br>", unsafe_allow_html = True)



# Primary Filters:


with st.sidebar:


	st.write("<h1><center>Filters</center></h1>", unsafe_allow_html = True)


	st.write("<br>", unsafe_allow_html = True)



	# Year Slider


	years = list(data.financial_year.unique())


	years.sort()


	sel_year = st.slider(


		"Select Year Range", 


		years[0], 


		years[-1],


		(years[0], years[1])


		)



	data = data[

	data.financial_year.isin(


		[i for i in range(sel_year[0], sel_year[-1] + 1)]


		)

	]



	st.write("<br>", unsafe_allow_html = True)



	# Quarter Multi Select


	quarts = ['All', 'Q1', 'Q2', 'Q3', 'Q4']



	sel_quarter = st.multiselect(


		'Select Quarter', 


		quarts, 


		default = quarts[quarts.index('All')]


		)



	if 'All' in sel_quarter:


		data = data[data.quarter.isin(quarts[1::1])]


	elif len(sel_quarter) == 0:


		data = data[data.quarter.isin(quarts[1::1])]



	else:


		data = data[data.quarter.isin(sel_quarter)]



	st.write("<br>", unsafe_allow_html = True)



	# Month Multi Select


	months_list = ['All']


	months = list(data.month.unique())


	months.sort()


	months_list.extend(months)



	sel_month = st.multiselect(

		"Select Month", 

		months_list , 

		default = months_list[months_list.index('All')]


		)



	if 'All' in sel_month:


		data = data[data.month.isin(months)]


	elif len(sel_month) == 0:


		data = data[data.month.isin(months)]


	else:


		data = data[data.month.isin(sel_month)]



	st.write("<br>", unsafe_allow_html = True)




	# Circle Multi Select


	circles = data.circle.unique()


	circles.sort()


	circle_list = ['All']


	circle_list.extend(circles)


	sel_circle = st.multiselect(

		"Select Circle", 

		circle_list, 

		default = ['SIDDIPET', 'SURYAPET', 'WANAPARTHY', 'YADADRI'],


		help = "TSSPDCL uses the term 'Circle' as a synonym for 'District'."


		)




	if 'All' in sel_circle:



		data = data[data.circle.isin(circles)]



	elif len(sel_circle) == 0:


		data = data[data.circle.isin(circles)]


	else:
		

		data = data[data.circle.isin(sel_circle)]



	st.write("<br>", unsafe_allow_html = True)




	# Division Multi Select



	divisions = data.division.unique()


	divisions.sort()


	division_list = ['All']


	division_list.extend(divisions)


	sel_div = st.multiselect(

		"Select Division", 

		division_list, 

		default = division_list[division_list.index('All')]


		)




	if 'All' in sel_div:



		data = data[data.division.isin(divisions)]



	elif len(sel_div) == 0:


		data = data[data.division.isin(divisions)]


	else:
		

		data = data[data.division.isin(sel_div)]



	st.write("<br>", unsafe_allow_html = True)



	# Sub Division Multi Select


	subdivisions = data.subdivision.unique()


	subdivisions.sort()


	subdivision_list = ['All']


	subdivision_list.extend(subdivisions)


	sel_subdiv = st.multiselect(

		"Select Sub-Division", 

		subdivision_list, 

		default = subdivision_list[subdivision_list.index('All')]


		)




	if 'All' in sel_subdiv:



		data = data[data.subdivision.isin(subdivisions)]



	elif len(sel_subdiv) == 0:


		data = data[data.subdivision.isin(subdivisions)]


	else:
		

		data = data[data.subdivision.isin(sel_subdiv)]



	st.write("<br>", unsafe_allow_html = True)



	# Section Multi Select



	sections = data.section.unique()


	sections.sort()


	section_list = ['All']


	section_list.extend(sections)


	sel_section = st.multiselect(

		"Select Section", 

		section_list, 

		default = section_list[section_list.index('All')]


		)




	if 'All' in sel_section:



		data = data[data.section.isin(sections)]



	elif len(sel_section) == 0:


		data = data[data.section.isin(sections)]


	else:
		

		data = data[data.section.isin(sel_section)]



	st.write("<br>", unsafe_allow_html = True)




# Plotly Line Charts Depicting Units Consumption by Load



# Chart Title


st.write('''<h2><center><b>Energy Consumption by Load Line Chart</b></center></h2>''', unsafe_allow_html = True)


st.write("<br><br>", unsafe_allow_html = True)



viz_df = data



# A. Sum

#	1. By Month & Year

#	2. By Quarter & Financial Year

# 	3. By Financial Year


# B. Average


#	1. By Quarter & Financial Year


#	2. By Financial Year






def get_yr_qtr_sum_data(viz_df):



	viz_yr_quarter = viz_df.groupby(


		['quart', 'year_quarter', 'load']

		, 

		as_index = False

		).agg(

		{'units': np.sum}

		).sort_values(

		by = ['quart', 'year_quarter', 'load'], 

		ascending = [True, True, True]


		)



	# st.dataframe(viz_yr_quarter)



	unique_loads = list(viz_yr_quarter.load.unique())


	unique_loads.sort()


	loads_list = ['All']


	loads_list.extend(unique_loads)



	sel_load = st.multiselect(


		'Select Loads (in kW)', 


		loads_list, 


		# default = [7.0, 15.0, 18.0, 30.0],


		default = ['All'],


		key = 'yr_qtr_sum_sel'


		)


	selected_loads = []


	if 'All' in sel_load:


		selected_loads = unique_loads



	elif len(sel_load) == 0:


		selected_loads = unique_loads




	else:


		selected_loads = sel_load




	fig = go.Figure()




	for load in selected_loads:



		load_df = viz_yr_quarter[viz_yr_quarter.load == load]


		load_df['units_lag'] = load_df.units.shift(1)


		load_df['units_%_change'] = round(((load_df.units - load_df.units_lag) / (load_df.units_lag)) * 100, 1)


		load_df['units_%_change'] = load_df['units_%_change'].apply(lambda x: 0.0 if 'nan' in str(x).lower() else x)


		fig.add_trace(


			go.Scatter(


				x = load_df.quart,


				y = load_df.units,


				mode = 'lines+markers',


				name = str(round(load, 1)) + " kW",


				customdata = load_df


				)


			)




	hovertemp = "<br><br>".join(


		[

			"<b>%{customdata[1]}</b>",


			"<b>%{y:.2s} Units Consumed</b>",


			"<b>%{customdata[5]:.1f}% vs Previous Quarter</b>"

		]


		)


	fig.update_traces(hovertemplate = hovertemp) 



	fig.update_yaxes(title = 'Units Consumed')



	# fig.update_xaxes(autorange = 'reversed')



	fig.update_xaxes(


		tickmode = 'array', 


		tickvals = viz_yr_quarter.quart.values,


		ticktext = viz_yr_quarter.year_quarter.values


		)



	fig.update_layout(showlegend = False)



	return st.plotly_chart(fig, use_container_width = True)








def get_yr_qtr_avg_data(viz_df):



	viz_yr_quarter = viz_df.groupby(


		['quart', 'year_quarter', 'load']

		, 

		as_index = False

		).agg(

		{'units': np.mean}

		).sort_values(

		by = ['quart', 'year_quarter', 'load'], 

		ascending = [True, True, True]


		)



	# st.dataframe(viz_yr_quarter)



	unique_loads = list(viz_yr_quarter.load.unique())


	unique_loads.sort()


	loads_list = ['All']


	loads_list.extend(unique_loads)



	sel_load = st.multiselect(


		'Select Loads (in kW)', 


		loads_list, 


		# default = [7.0, 15.0, 18.0, 30.0],


		default = ['All'],


		key = 'yr_qtr_avg_sel'


		)


	selected_loads = []


	if 'All' in sel_load:


		selected_loads = unique_loads



	elif len(sel_load) == 0:


		selected_loads = unique_loads




	else:


		selected_loads = sel_load



	fig = go.Figure()




	for load in selected_loads:



		load_df = viz_yr_quarter[viz_yr_quarter.load == load]


		load_df['units_lag'] = load_df.units.shift(1)


		load_df['units_%_change'] = round(((load_df.units - load_df.units_lag) / (load_df.units_lag)) * 100, 1)


		load_df['units_%_change'] = load_df['units_%_change'].apply(lambda x: 0.0 if 'nan' in str(x).lower() else x)



		fig.add_trace(


			go.Scatter(


				x = load_df.quart,


				y = load_df.units,


				mode = 'lines+markers',


				name = str(round(load, 1)) + " kW",


				customdata = load_df


				)


			)




	hovertemp = "<br><br>".join(


		[

			"<b>%{customdata[1]}</b>",


			"<b>%{y:.2s} Units Consumed Monthly On Average</b>",


			"<b>%{customdata[5]:.1f}% vs Previous Quarter</b>"

		]


		)


	fig.update_traces(hovertemplate = hovertemp) 



	fig.update_yaxes(title = 'Units Consumed')



	# fig.update_xaxes(autorange = 'reversed')



	fig.update_xaxes(


		tickmode = 'array', 


		tickvals = viz_yr_quarter.quart.values,


		ticktext = viz_yr_quarter.year_quarter.values


		)



	fig.update_layout(showlegend = False)



	return st.plotly_chart(fig, use_container_width = True)








def get_yr_month_sum_data(viz_df):



	viz_yr_month = viz_df.groupby(

		['year_month', 'load'],

		as_index = False

		).agg(

		{'units': np.sum}

		).sort_values(

		by = ['year_month', 'load'], 

		ascending = [True, True]

		)


	# st.dataframe(viz_yr_month.head(5))


	unique_loads = list(viz_yr_month.load.unique())


	unique_loads.sort()


	# Loads Filter


	loads_list = ['All']


	loads_list.extend(unique_loads)


	sel_load = st.multiselect(


		'Select Loads (in kW)', 


		loads_list, 


		# default = [7.0, 15.0, 18.0, 30.0],


		default = ['All'],


		key = 'yr_month_sum_sel'


	)


	selected_loads = []


	if 'All' in sel_load:


		selected_loads = unique_loads


	elif len(sel_load) == 0:


		selected_loads = unique_loads


	else:


		selected_loads = sel_load



	fig = go.Figure()



	for load in selected_loads:



		load_df = viz_yr_month[viz_yr_month.load == load]


		load_df['units_lag'] = load_df.units.shift(1)


		load_df['units_%_change'] = round(((load_df.units - load_df.units_lag) / (load_df.units_lag)) * 100, 1)


		load_df['units_%_change'] = load_df['units_%_change'].apply(lambda x: 0.0 if 'nan' in str(x).lower() else x)




		fig.add_trace(


			go.Scatter(

				x = load_df.year_month,


				y = load_df.units,


				mode = 'lines+markers',


				name = f"{str(round(load, 1)) + ' ' + 'kW'}",


				customdata = load_df


				)




			)




	hovertemp = "<br><br>".join(

		[

			"<b>%{x}</b>",



			"<b>%{y:.2s} Units Consumed</b>",


			"<b>%{customdata[4]:.1f}% vs Previous Month</b>"



		]


		)



	fig.update_traces(hovertemplate = hovertemp)



	# fig.update_traces(text = viz_yr_month[viz_yr_month.load == load].units)



	# fig.update_traces(texttemplate = '%{text:.2s}', textposition = 'middle center')



	fig.update_yaxes(title = 'Units Consumed')



	fig.update_layout(showlegend = False)




	return st.plotly_chart(fig, use_container_width = True)








def get_yr_sum_data(viz_df):


	viz_yr = viz_df.groupby(

		['financial_year', 'load'],

		as_index = False

		).agg(

		{'units': np.sum}

		).sort_values(

		by = ['financial_year', 'load'], 

		ascending = [True, True]

		)


	financial_years = pd.Series(viz_yr.financial_year.unique())



	financial_years = pd.concat(

		[financial_years, pd.Series([financial_years.max() + 1])], 

		axis = 0

		)


	financial_years = financial_years.apply(lambda x: str(x) + '-' + str(x + 1) if x != financial_years.iloc[-1] else 'None')


	financial_years = financial_years.values[:-1:1]


	financial_years_dict = {k:v for k, v in zip([str(i) for i in viz_yr.financial_year.unique()], list(financial_years))}


	viz_yr['fyear'] = viz_yr.financial_year.apply(lambda x: financial_years_dict[str(x)])


	# st.dataframe(viz_yr.head(5))


	unique_loads = list(viz_yr.load.unique())


	unique_loads.sort()


	# Loads Filter


	loads_list = ['All']


	loads_list.extend(unique_loads)


	sel_load = st.multiselect(


		'Select Loads (in kW)', 


		loads_list, 


		# default = [7.0, 15.0, 18.0, 30.0],



		default = ['All'],





		key = 'yr_sum_sel'


	)


	selected_loads = []


	if 'All' in sel_load:


		selected_loads = unique_loads


	elif len(sel_load) == 0:


		selected_loads = unique_loads


	else:


		selected_loads = sel_load




	fig = go.Figure()



	for load in selected_loads:



		load_df = viz_yr[viz_yr.load == load]


		load_df['units_lag'] = load_df.units.shift(1)


		load_df['units_%_change'] = round(((load_df.units - load_df.units_lag) / (load_df.units_lag)) * 100, 1)


		load_df['units_%_change'] = load_df['units_%_change'].apply(lambda x: 0.0 if 'nan' in str(x).lower() else x)




		fig.add_trace(


			go.Scatter(

				x = load_df.financial_year,


				y = load_df.units,


				mode = 'lines+markers',


				name = f"{str(round(load, 1)) + ' ' + 'kW'}",


				customdata = load_df


				)




			)




	hovertemp = "<br><br>".join(

		[

			"<b>%{x}</b>",



			"<b>%{y:.2s} Units Consumed</b>",



			"<b>%{customdata[5]:.1f}% vs Previous Financial Year</b>"



		]


		)



	fig.update_traces(hovertemplate = hovertemp)



	# fig.update_traces(text = viz_yr_month[viz_yr_month.load == load].units)



	# fig.update_traces(texttemplate = '%{text:.2s}', textposition = 'middle center')



	fig.update_yaxes(title = 'Units Consumed')


	fig.update_xaxes(


		tickmode = 'array', 


		tickvals = viz_yr.financial_year.values,


		ticktext = viz_yr.fyear.values


		)




	fig.update_layout(showlegend = False)




	return st.plotly_chart(fig, use_container_width = True)








def get_yr_avg_data(viz_df):


	viz_yr = viz_df.groupby(

		['financial_year', 'load'],

		as_index = False

		).agg(

		{'units': np.mean}

		).sort_values(

		by = ['financial_year', 'load'], 

		ascending = [True, True]

		)


	financial_years = pd.Series(viz_yr.financial_year.unique())



	financial_years = pd.concat(

		[financial_years, pd.Series([financial_years.max() + 1])], 

		axis = 0

		)


	financial_years = financial_years.apply(lambda x: str(x) + '-' + str(x + 1) if x != financial_years.iloc[-1] else 'None')


	financial_years = financial_years.values[:-1:1]


	financial_years_dict = {k:v for k, v in zip([str(i) for i in viz_yr.financial_year.unique()], list(financial_years))}


	viz_yr['fyear'] = viz_yr.financial_year.apply(lambda x: financial_years_dict[str(x)])


	# st.dataframe(viz_yr.head(5))


	unique_loads = list(viz_yr.load.unique())


	unique_loads.sort()


	# Loads Filter


	loads_list = ['All']


	loads_list.extend(unique_loads)


	sel_load = st.multiselect(


		'Select Loads (in kW)', 


		loads_list, 


		# default = [7.0, 15.0, 18.0, 30.0],

		default = ['All'],


		key = 'yr_avg_sel'


	)


	selected_loads = []


	if 'All' in sel_load:


		selected_loads = unique_loads


	elif len(sel_load) == 0:


		selected_loads = unique_loads


	else:


		selected_loads = sel_load




	fig = go.Figure()



	for load in selected_loads:



		load_df = viz_yr[viz_yr.load == load]


		load_df['units_lag'] = load_df.units.shift(1)


		load_df['units_%_change'] = round(((load_df.units - load_df.units_lag) / (load_df.units_lag)) * 100, 1)


		load_df['units_%_change'] = load_df['units_%_change'].apply(lambda x: 0.0 if 'nan' in str(x).lower() else x)







		fig.add_trace(


			go.Scatter(


				x = load_df.financial_year,


				y = load_df.units,


				mode = 'lines+markers',


				name = f"{str(round(load, 1)) + ' ' + 'kW'}",


				customdata = load_df


				)




			)




	hovertemp = "<br><br>".join(

		[

			"<b>%{x}</b>",



			"<b>%{y:.2s} Units Consumed Monthly On An Average</b>",



			"<b>%{customdata[5]:.1f}% vs Previous Financial Year</b>"



		]


		)



	fig.update_traces(hovertemplate = hovertemp)



	# fig.update_traces(text = viz_yr_month[viz_yr_month.load == load].units)



	# fig.update_traces(texttemplate = '%{text:.2s}', textposition = 'middle center')



	fig.update_yaxes(title = 'Units Consumed')


	fig.update_xaxes(


		tickmode = 'array', 


		tickvals = viz_yr.financial_year.values,


		ticktext = viz_yr.fyear.values


		)




	fig.update_layout(showlegend = False)




	return st.plotly_chart(fig, use_container_width = True)





# get_yr_month_sum_data(viz_df)



# get_yr_qtr_avg_data(viz_df)



# get_yr_sum_data(viz_df)


# get_yr_avg_data(viz_df)




units_consumption_chart_dict = {"Sum": [


'Month & Year', 


'Quarter & Financial Year', 


'Financial Year'


], 


"Average": [



'Quarter & Financial Year', 



'Financial Year'


]


}



# Line Chart Type Session State



if 'line_chart_type' not in st.session_state:

	st.session_state.line_chart_type = 'Sum'





# Line Chart Metric Session State



if 'line_chart_metric' not in st.session_state:


	st.session_state.line_chart_metric = units_consumption_chart_dict[st.session_state.line_chart_type]



# Line Chart Type Callback Function


def change_type_of_line_chart():


	if st.session_state.line_chart_type:


		st.session_state.line_chart_type = st.session_state.new_line_chart_type


	else:

		pass



# Line Chart Metric Callback Function


def change_consumption_line_chart_metric():


	if st.session_state.line_chart_metric:


		st.session_state.line_chart_metric = st.session_state.new_line_chart_metric




# Line Chart Buttons



lccol1, lccol2 = st.columns(2)



with lccol1:



	type_of_line_chart = st.selectbox(


		"Select Aggregation", 


		list(units_consumption_chart_dict.keys()),


		on_change = change_type_of_line_chart,


		key = 'new_line_chart_type'


		)




with lccol2:



	line_chart_metric = st.selectbox(


		'Select Time Series', 


		units_consumption_chart_dict[st.session_state.line_chart_type],


		on_change = change_consumption_line_chart_metric,


		key = 'new_line_chart_metric'


		)




st.write("<br>", unsafe_allow_html = True)





if type_of_line_chart == 'Sum':


	if  line_chart_metric == 'Month & Year':


		get_yr_month_sum_data(viz_df)



	elif line_chart_metric == 'Quarter & Financial Year':


		get_yr_qtr_sum_data(viz_df)



	elif line_chart_metric == 'Financial Year':


		get_yr_sum_data(viz_df)


	else:


		pass



elif type_of_line_chart == 'Average':


	if line_chart_metric == 'Quarter & Financial Year':


		get_yr_qtr_avg_data(viz_df)



	elif line_chart_metric == 'Financial Year':


		get_yr_avg_data(viz_df)


	else:

		pass 




else:

	pass








# Folium Maps


# Title



st.write("<br>", unsafe_allow_html = True)



st.write('''<h2><center>Interactive Map</center></h2>''', unsafe_allow_html = True)



st.write("<br>", unsafe_allow_html = True)



map_df = data





def get_sum_df():


	# Units Sum:



	map_grp_sum_df = map_df.groupby(

		['address_latitude', 'address_longitude', 'address', 'load', 'circle', 'district']

		, as_index = False

		).agg(

		{"units": np.sum}


		)


	map_grp_sum_df.columns = ['address_latitude', 'address_longitude', 'address', 'load', 'circle', 'district', 'total_units_consumed']



	sum_quartiles = pd.DataFrame(map_grp_sum_df.total_units_consumed.describe())


	sum_cats = []


	for val in map_grp_sum_df.total_units_consumed.values:


		if (val >= sum_quartiles.loc['min', 'total_units_consumed']) and (val < sum_quartiles.loc['25%', 'total_units_consumed']):



			sum_cats.append('min-25')



		elif (val >= sum_quartiles.loc['25%', 'total_units_consumed']) and (val < sum_quartiles.loc['50%', 'total_units_consumed']):


			sum_cats.append('25-50')


		elif (val >= sum_quartiles.loc['50%', 'total_units_consumed']) and (val < sum_quartiles.loc['75%', 'total_units_consumed']):


			sum_cats.append('50-75')



		elif (val >= sum_quartiles.loc['75%', 'total_units_consumed']) and (val < sum_quartiles.loc['max', 'total_units_consumed']):


			sum_cats.append('75-max')



		else: 


			sum_cats.append('max')



	map_grp_sum_df['category'] = sum_cats



	sum_colors = []



	for cat in map_grp_sum_df.category.values:


		if cat == 'min-25':


			sum_colors.append('gray')


		elif cat == '25-50':


			sum_colors.append('lightgray')


		elif cat == '50-75':


			sum_colors.append('orange')


		elif cat == '75-max':


			sum_colors.append('lightred')


		else:


			sum_colors.append('red')



	map_grp_sum_df['color'] = sum_colors


	map_grp_sum_df = map_grp_sum_df.reset_index(drop = True)



	return map_grp_sum_df





def get_mean_df():


	# Units Mean:



	map_grp_mean_df = map_df.groupby(


		['address_latitude', 'address_longitude', 'address', 'load', 'circle', 'district']

		, as_index = False

		).agg(

		{"units": np.mean}


		)


	map_grp_mean_df.columns = ['address_latitude', 'address_longitude', 'address', 'load', 'circle', 'district', 'avg_units_consumed']



	mean_quartiles = pd.DataFrame(map_grp_mean_df.avg_units_consumed.describe())



	mean_cats = []


	for val in map_grp_mean_df.avg_units_consumed.values:


		if (val >= mean_quartiles.loc['min', 'avg_units_consumed']) and (val < mean_quartiles.loc['25%', 'avg_units_consumed']):



			mean_cats.append('min-25')



		elif (val >= mean_quartiles.loc['25%', 'avg_units_consumed']) and (val < mean_quartiles.loc['50%', 'avg_units_consumed']):


			mean_cats.append('25-50')


		elif (val >= mean_quartiles.loc['50%', 'avg_units_consumed']) and (val < mean_quartiles.loc['75%', 'avg_units_consumed']):


			mean_cats.append('50-75')



		elif (val >= mean_quartiles.loc['75%', 'avg_units_consumed']) and (val < mean_quartiles.loc['max', 'avg_units_consumed']):


			mean_cats.append('75-max')



		else: 


			mean_cats.append('max')



	map_grp_mean_df['category'] = mean_cats



	mean_colors = []



	for cat in map_grp_mean_df.category.values:


		if cat == 'min-25':


			mean_colors.append('gray')


		elif cat == '25-50':


			mean_colors.append('lightgray')


		elif cat == '50-75':


			mean_colors.append('orange')


		elif cat == '75-max':


			mean_colors.append('lightred')


		else:


			mean_colors.append('red')



	map_grp_mean_df['color'] = mean_colors



	map_grp_mean_df = map_grp_mean_df.reset_index(drop = True)



	return map_grp_mean_df




# Folium Map Funcs



# Total Units Consumed Map Func



def get_sum_map(map_grp_sum_df):



		unique_loads = list(map_grp_sum_df.load.unique())


		unique_loads.sort()



		loads_list = ['All']


		loads_list.extend(unique_loads)



		sel_loads = st.multiselect(


			"Select Loads (in kW)", 


			loads_list, 


			default = [loads_list[loads_list.index('All')]]



			)



		if 'All' in sel_loads:


			map_grp_sum_df = map_grp_sum_df[map_grp_sum_df.load.isin(unique_loads)]



		elif len(sel_loads) == 0:


			map_grp_sum_df = map_grp_sum_df[map_grp_sum_df.load.isin(unique_loads)]



		else:


			map_grp_sum_df = map_grp_sum_df[map_grp_sum_df.load.isin(sel_loads)]




		map_grp_sum_df = map_grp_sum_df.reset_index(drop = True)



		st.write("<br>", unsafe_allow_html = True)



		map = folium.Map(


			[17.39563665403449, 78.46520083886992], 


			zoom_start = 8.25, 


			tiles = 'cartodbpositron'

		)




		for i in range(len(map_grp_sum_df)):


			ttip = map_grp_sum_df.loc[i, 'address'].split(', ')[:-1:1]


			ttip.append(map_grp_sum_df.loc[i, 'district'].strip().upper().replace('_', ' '))


			tooltip = []


			print([tooltip.append(i) for i in ttip if i not in tooltip])



			folium.Marker(

				[

				map_grp_sum_df.loc[i]['address_latitude'], 


				map_grp_sum_df.loc[i]['address_longitude']

				],

				icon = folium.Icon(


					color = map_grp_sum_df.loc[i]['color'],


					icon = 'location-pin',


					prefix = 'fa'


					),


				tooltip = f"<b>{', '.join(tooltip)}<br><br>{str(round(map_grp_sum_df.loc[i, 'load'], 1)) + ' kW CHARGER'}<br><br>{str(int(map_grp_sum_df.loc[i, 'total_units_consumed'])) + ' UNITS CONSUMED'}</b>"


				).add_to(map)




		ta_features = [feature for feature in ta_districts['features'] if feature['properties']['Dist_Name'] in map_grp_sum_df.district.unique()]


		ta_districts['features'] = ta_features



		folium.GeoJson(ta_districts).add_to(map)




		# st_folium(map, use_container_width = True)


		folium_static(map, width = 1100)



		st.write("<br>", unsafe_allow_html = True)



		st.write("<h5><center><u>Please Note </u><br><br>The color of the location pins for EV chargers with high unit consumption will be redder than those with low unit consumption.<br><br>Similarly, the color of the location pins for EV chargers with low unit consumption will be darker on the gray side than those with high unit consumption.<br><br>Most importantly, also note that 1 unit equals 1 kWh.</center></h5>", unsafe_allow_html = True)



		st.write("<br>", unsafe_allow_html = True)





# Mean Units Consumed Map Func



def get_avg_map(map_grp_mean_df):



		unique_loads = list(map_grp_mean_df.load.unique())


		unique_loads.sort()



		loads_list = ['All']


		loads_list.extend(unique_loads)



		sel_loads = st.multiselect(


			"Select Loads (in kW)", 


			loads_list, 


			default = [loads_list[loads_list.index('All')]]



			)



		if 'All' in sel_loads:


			map_grp_mean_df = map_grp_mean_df[map_grp_mean_df.load.isin(unique_loads)]



		elif len(sel_loads) == 0:


			map_grp_mean_df = map_grp_mean_df[map_grp_mean_df.load.isin(unique_loads)]



		else:


			map_grp_mean_df = map_grp_mean_df[map_grp_mean_df.load.isin(sel_loads)]




		map_grp_mean_df = map_grp_mean_df.reset_index(drop = True)


		st.write("<br>", unsafe_allow_html = True)


		map = folium.Map(

			[17.39563665403449, 78.46520083886992], 

			zoom_start = 8.25, 

			tiles = 'cartodbpositron'

		)




		for i in range(len(map_grp_mean_df)):


			ttip = map_grp_mean_df.loc[i, 'address'].split(', ')[:-1:1]


			ttip.append(map_grp_mean_df.loc[i, 'district'].strip().upper().replace("_", ' '))


			tooltip = []


			print([tooltip.append(i) for i in ttip if i not in tooltip])


			# tooltip.append('TELANGANA')



			folium.Marker(

				[

				map_grp_mean_df.loc[i]['address_latitude'], 

				map_grp_mean_df.loc[i]['address_longitude']

				],

				icon = folium.Icon(

					color = map_grp_mean_df.loc[i]['color'],

					icon = 'location-pin',

					prefix = 'fa'

					),


				tooltip = f"<b>{', '.join(tooltip)}<br><br>{str(round(map_grp_mean_df.loc[i, 'load'], 1)) + ' kW CHARGER'}<br><br>{str(int(map_grp_mean_df.loc[i, 'avg_units_consumed'])) + ' UNITS CONSUMED MONTHLY ON AVERAGE'}</b>"


				).add_to(map)




		ta_features = [feature for feature in ta_districts['features'] if feature['properties']['Dist_Name'] in map_grp_mean_df.district.unique()]


		ta_districts['features'] = ta_features


		folium.GeoJson(ta_districts).add_to(map)



		# st_folium(map, use_container_width = True)


		folium_static(map, width = 1100)



		st.write("<br>", unsafe_allow_html = True)



		st.write("<h5><center><u>Please Note </u><br><br>The color of the location pins for EV chargers with high unit consumption will be redder than those with low unit consumption.<br><br>Similarly, the color of the location pins for EV chargers with low unit consumption will be darker on the gray side than those with high unit consumption.<br><br>Most importantly, also note that 1 unit equals 1 kWh.</center></h5>", unsafe_allow_html = True)



		st.write("<br>", unsafe_allow_html = True)





# Map Buttons


if 'map_metric' not in st.session_state:


	st.session_state.map_metric = 'Sum'




def change_map_metric():


	if st.session_state.map_metric:


		st.session_state.map_metric = st.session_state.new_map_metric





select_map_metric = st.selectbox(


	'Select Aggregation', 


	['Sum', 'Average'], 


	on_change = change_map_metric, 


	key = 'new_map_metric'


	)



st.write("<br>", unsafe_allow_html = True)



if select_map_metric == 'Sum':


	get_sum_map(get_sum_df())



else:

	get_avg_map(get_mean_df())




# Data Source



st.write("<br><br><br><br>", unsafe_allow_html = True)




st.write('''<footer class="css-164nlkn egzxvld1"><center><p>Data Source: <a href="https://data.telangana.gov.in/" target="_blank" class="css-1vbd788 egzxvld2">data.telangana.gov.in</a></p></center></footer>''', unsafe_allow_html = True)
