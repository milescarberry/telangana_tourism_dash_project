import numpy as np

import pandas as pd

# import matplotlib.pyplot as plt

# import seaborn as sns

# sns.set_context("paper", font_scale = 1.4)

from plotly import express as exp, graph_objects as go, io as pio

pio.templates.default == 'ggplot2'

from plotly.subplots import make_subplots

import pickle

import datetime as dt

from pandas_utils.pandas_utils_2 import *

import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)



st.set_page_config(


	page_title = "Page Title", 


	layout = 'wide'


	)






@st.cache_data
def get_datasets():


	# Get datasets 

	pass 




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





# Sidebar Filters




with st.sidebar:


	# Some Dataset Filters








# Do your work here










# Footer Section



# Mention Data Source



st.write("<br><br><br><br>", unsafe_allow_html = True)




st.write(

	'''<footer class="css-164nlkn egzxvld1"><center><p>Data Source: <a href="https://data.telangana.gov.in/" target="_blank" class="css-1vbd788 egzxvld2">data.telangana.gov.in</a></p></center></footer>''', 


	unsafe_allow_html = True


	)




