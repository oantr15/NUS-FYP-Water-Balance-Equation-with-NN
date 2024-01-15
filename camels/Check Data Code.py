## Import dependent libraries
import os
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import detrend

basin_id = '02299950' # The basin_id can be changed to any 8-digit basin id contained in the basin_list.txt
#working_path = os.path.join(os.getcwd(),'GW_data',basin_id + '.csv')
#working_pather = os.path.join(os.getcwd(),'GW_data',basin_id + '.txt')
#hydrodata = pd.read_csv(working_path)
GW_data = pd.read_csv(r'C:\Users\Chin Seng\Desktop\CE4104\Project\Code\camels\GW_data\02092500.txt')

print(GW_data)