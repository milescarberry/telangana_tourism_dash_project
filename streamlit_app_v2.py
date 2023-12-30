import numpy as np

import pandas as pd

# import matplotlib.pyplot as plt

# import seaborn as sns

# sns.set_context("paper", font_scale = 1.4)

from plotly import express as exp, graph_objects as go, io as pio

from plotly.subplots import make_subplots

import streamlit as st

import pickle

import datetime as dt

import time

from pandas_utils.pandas_utils_2 import *

import warnings



st.set_page_config(


	page_title = "Page Title", 


	layout = 'wide'


	)


warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)

pio.templates.default = 'ggplot2'


# Reduce Top Padding In The Main Page & The Sidebar


st.markdown(" <style> div[class^='st-emotion-cache-16txtl3'] { padding-top: 2rem; } </style> ", unsafe_allow_html=True)


st.markdown(" <style> div[class^='block-container'] { padding-top: 2rem; } </style> ", unsafe_allow_html=True)




# Get Datasets Func


@st.cache_data
def get_datasets():


	# Get datasets 

	pass 




# Dashboard Title


# st.write(

# 	"<h1><center>Title</center></h1>",


# 	unsafe_allow_html = True


# 	)




# st.write("<br><br>", unsafe_allow_html = True)



# # Some More Text


# st.write(

# 	"<h5><center>Some more text comes here.</center></h5>", 

# 	unsafe_allow_html = True

# 	)



# st.write("<br><br>", unsafe_allow_html = True)





# Sidebar Filters




with st.sidebar:


	# Some Dataset Filters


	pass






# Columns


col1, col2, col3 = st.columns([1, 2, 1])            # Column size ratios



with col1:


	st.markdown("# Welcome to my app!")

	st.write("<br>", unsafe_allow_html = True)

	st.markdown("Here is some more text.")


with col2:



	def photo_uploader_callback_func():


		if st.session_state.new_upload is not None:


			upload_photo_progress_bar = st.progress(0)


			for i in range(100):


				time.sleep(0.023)


				upload_photo_progress_bar.progress(i + 1)



			st.success("Photo uploaded successfully!")



		else:


			st.write("Please upload a photo.")





	uploaded_photo = st.file_uploader(

		"Upload a Photo", 

		on_change = photo_uploader_callback_func, 

		key = 'new_upload'

		)


	# camera_photo = st.camera_input("Take a Picture")





with col3:


	st.metric(label = "Temperature", value = "37 °C", delta = "-1.5 °C")





st.write("<br><br>", unsafe_allow_html = True)



with st.expander("Click here to know more."):


	st.write("Here is some more info.")


	if uploaded_photo is not None:


		st.image(uploaded_photo)



	else:


		st.write("No photo uploaded.")






# Footer Section



# Mention Data Source



st.write("<br><br><br><br>", unsafe_allow_html = True)




st.write(

	'''<footer class="css-164nlkn egzxvld1"><center><p>Data Source: <a href="https://data.telangana.gov.in/" target="_blank" class="css-1vbd788 egzxvld2">data.telangana.gov.in</a></p></center></footer>''', 


	unsafe_allow_html = True


	)




