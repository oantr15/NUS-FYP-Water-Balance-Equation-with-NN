"""
This file is part of the accompanying code to our manuscript:

Jiang S., Zheng Y., & Solomatine D.. (2020) Improving AI system awareness of geoscience knowledge: Symbiotic integration of physical approaches and deep learning. Geophysical Research Letters, 47, e2020GL088229. https://doi.org/10.1029/2020GL088229

Copyright (c) 2020 Shijie Jiang. All rights reserved.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import os
import pandas as pd
from datetime import datetime
import glob

class DataPathWays():
    def __init__(self, working_path):
        self.working_path = working_path
        
    def basin_id_finder(self, working_path):
        filepath = os.path.join(working_path,'camels', 'camels_03', 'GW_data')
        result = glob.glob(filepath + "/*.csv")
        all_basin_names = [os.path.basename(path)[:8] for path in result]
        return all_basin_names
        

class DataforIndividual():
    def __init__(self, working_path, basin_id):
        self.working_path = working_path
        self.basin_id = basin_id

    def check_validation(self, basin_list, basin_id):
        assert isinstance(basin_id, str), "The basin ID should be a string"
        assert (len(basin_id) == 8 and basin_id.isdigit()), "Basin ID can only be represented by 8 digits"
        assert (basin_id in basin_list.values), "Please confirm the basin specified is in basin_list.txt"

    def ground_water_data(self, working_path, huc_id, basin_id):
        gw_path = os.path.join(working_path,'camels', 'camels_03', 'GW_data', basin_id + '.csv')
        gw_data = pd.read_csv(gw_path)
        gw_data['date'] = pd.to_datetime(gw_data['date'], dayfirst=True, errors='ignore')
        return gw_data
    
    def load_force_data(self, working_path, huc_id, basin_id):
        forcing_path = os.path.join(working_path, 'camels', 'camels_03', 'basin_mean_forcing',
                                    basin_id + '_lump_cida_forcing_leap.txt')
        force_data =  pd.read_csv(forcing_path, sep="\s+|;|:", header=0, skiprows=3, engine='python')
        force_data.rename(columns={"Mnth": "Month"}, inplace=True)
        force_data['date'] = pd.to_datetime(force_data[['Year', 'Month', 'Day']], dayfirst=True, errors='ignore')
        force_data['dayl(day)'] =  force_data['dayl(s)']/ 86400
        with open(forcing_path, 'r') as fp:
            content = fp.readlines()
            area = int(content[2])
        return area, force_data
    
    def load_flow_data(self, working_path, huc_id, basin_id, area):
        flow_path = os.path.join(working_path, 'camels', 'camels_03','usgs_streamflow',
                                 basin_id + '_streamflow_qc.txt')
        flow_data = pd.read_csv(flow_path, sep="\s+", names=['Id', 'Year', 'Month', 'Day', 'Q', 'QC'],
                                header=None, engine='python')
        flow_data['date'] = pd.to_datetime(flow_data[['Year', 'Month', 'Day']], dayfirst=True, errors='ignore')
        flow_data['flow(mm)'] = 28316846.592 * flow_data['Q'] * 86400 / (area * 10 ** 6)
        return flow_data

    def merge_data(self, working_path, huc_id, basin_id, gw_data, missing_cols, force_data, flow_data):
        for col in missing_cols:
            if col in flow_data.columns:
                merge_data = flow_data[['date',col]]
            elif col in force_data.columns:
                merge_data = force_data[['date',col]]
            gw_data = pd.merge(gw_data, merge_data, on='date')    
        final_df = gw_data[['date','prcp(mm/day)', 'tmean(C)', 'dayl(day)', 'srad(W/m2)', 'vp(Pa)', 'flow(mm)','GW(feet)']]
        return final_df
    
    def load_data(self):
        basin_list = pd.read_csv(os.path.join(self.working_path, 'basin_list.txt'),
                                 sep='\t', header=0, dtype={'HUC': str, 'BASIN_ID': str})
        self.check_validation(basin_list, self.basin_id)
        huc_id = basin_list[basin_list['BASIN_ID'] == self.basin_id]['HUC'].values[0]
        gw_data = self.ground_water_data(self.working_path, huc_id, self.basin_id)
        
        data_cols = ['prcp(mm/day)', 'tmean(C)', 'dayl(day)', 'srad(W/m2)', 'vp(Pa)', 'flow(mm)','GW(feet)']
        missing_col = [col for col in data_cols if col not in gw_data.columns]
        
        area, force_data = self.load_force_data(self.working_path, huc_id, self.basin_id)
        flow_data = self.load_flow_data(self.working_path, huc_id, self.basin_id, area)
        
        if missing_col != []:
            gw_data = self.merge_data(self.working_path, huc_id, self.basin_id, gw_data, missing_col,
                                      force_data, flow_data)

        final_gw_data = gw_data[(gw_data['date'] >= datetime(1980, 10, 1)) &
                                  (gw_data['date'] <= datetime(2010, 9, 30))]
        final_gw_data.sort_values(by='date', inplace = True) 
        final_gw_data['GW(feet)'] = final_gw_data['GW(feet)']
        
        final_gw_data = final_gw_data.set_index('date')
        print('Data in basin #{} at huc #{} has been successfully loaded.'.format(self.basin_id, huc_id))
        return final_gw_data
