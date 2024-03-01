## Import dependent libraries
import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from keras.models import Model
from keras.layers import Input, Concatenate, LSTM, Dense
from keras import optimizers, callbacks, regularizers
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

## Import libraries developed by this study
os.chdir(r'C:\Users\Chin Seng\Desktop\CE4104\Project\Code') #spyder issue, need re-route my current dir
from libs.hydrolayer import PRNNLayer, ScaleLayer
from libs.hydrodata import DataforIndividual, DataPathWays
from libs import hydroutils

## Ignore all the warnings
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_WARNINGS'] = '0'

#%%
####################
#   Basin set up   #
####################

#basin_id = '02296500' # The basin_id can be changed to any 8-digit basin id contained in the basin_list.txt
working_path = os.getcwd()
all_basins = DataPathWays(working_path).basin_id_finder(working_path) #get all basin_id and work with it

for basin_id in all_basins:
    hydrodata = DataforIndividual(working_path, basin_id).load_data()
    
    # Plot the data loaded for overview
    fig, [ax1, ax2, ax3, ax4, ax5, ax6, ax7] = plt.subplots(nrows=7, ncols=1, sharex='row', figsize=(15, 18))
    
    ax1.plot(hydrodata['prcp(mm/day)'])
    ax2.plot(hydrodata['tmean(C)'])
    ax3.plot(hydrodata['dayl(day)'])
    ax4.plot(hydrodata['srad(W/m2)'])
    ax5.plot(hydrodata['vp(Pa)'])
    ax6.plot(hydrodata['flow(mm)'])
    ax7.plot(hydrodata['GW(feet)'])
    
    ax1.set_title(f"Basin {basin_id}")
    ax1.set_ylabel("prcp(mm/day)")
    ax2.set_ylabel("tmean(C)")
    ax3.set_ylabel("dayl(day)")
    ax4.set_ylabel("srad(W/m2)")
    ax5.set_ylabel("vp(Pa)")
    ax6.set_ylabel("flow(mm)")
    ax7.set_ylabel('GW(feet)')
    
    # plt.savefig(os.path.join(os.getcwd(),'comparisons','Data for Basin #'+ basin_id + '.png'))
    
    ####################
    #  Period set up   #
    ####################
    
    training_start = '1980-10-01'
    training_end= '2000-09-30'
    
    # The REAL evaluation period is from '2000-10-01', while the model needs one-year of data for spinning up the model
    testing_start = '1999-10-01'
    testing_end= '2010-09-30'
    
    # Split data set to training_set and testing_set
    train_set = hydrodata[hydrodata.index.isin(pd.date_range(training_start, training_end))]
    test_set = hydrodata[hydrodata.index.isin(pd.date_range(testing_start, testing_end))]
        
    if test_set.shape[0] == 0 or train_set.shape[0] == 0:
        continue
    
    print(f"The training data set is from {training_start} to {training_end}, with a shape of {train_set.shape}")
    print(f"The testing data set is from {testing_start} to {testing_end}, with a shape of {test_set.shape}")

    def generate_train_test(train_set, test_set, wrap_length, wrap_size):
        train_x_np = train_set.values[:, :4] #change this for number of features
        train_y_np = train_set.values[:,-2:]
        test_x_np = test_set.values[:, :4] #change this for number of features
        test_y_np = test_set.values[:, -2:]
    
        wrap_number_train = (train_set.shape[0] - wrap_length) // wrap_size + 1
    
        train_x = np.empty(shape=(wrap_number_train, wrap_length, train_x_np.shape[1]))
        train_y = np.empty(shape=(wrap_number_train, wrap_length, train_y_np.shape[1]))
    
        test_x = np.expand_dims(test_x_np, axis=0)
        test_y = np.expand_dims(test_y_np, axis=0)
    
        for i in range(wrap_number_train):
            train_x[i, :, :] = train_x_np[i * wrap_size:(wrap_length + i * wrap_size), :]
            train_y[i, :, :] = train_y_np[i * wrap_size:(wrap_length + i * wrap_size), :]
    
        return train_x, train_y, test_x, test_y


    wrap_length = 2190  # It can be other values, but recommend this value should not be less than 5 years (1825 days).
    wrap_size = 365
    
    try:
        train_x, train_y, test_x, test_y = generate_train_test(train_set, test_set, 
                                                               wrap_length=wrap_length,
                                                               wrap_size=wrap_size)
    except: 
        continue
    print(f'The shape of train_x, train_y, test_x, and test_y after wrapping by {wrap_length} days are:')
    print(f'{train_x.shape}, {train_y.shape}, {test_x.shape}, and {test_y.shape}')

    def create_model(input_shape, input_size, model_type='hybrid'):
        # Create a Keras model.
        # -- input_shape: the shape of input, controlling the time sequence length of the P-RNN
        # -- seed: the random seed for the weights initialization of the 1D-CNN layers
        # -- num_filters: the number of filters for the 1D-CNN layer
        # -- model_type: can be 'hybrid', 'physical', or 'common'
    
        x_input = Input(shape=input_shape, name='Input')
    
        if model_type == 'hybrid':
            hydro_output = PRNNLayer(mode='normal', name='Hydro')(x_input)
            x_hydro = Concatenate(axis=-1,name='Concat')([x_input, hydro_output])
            x_scale = ScaleLayer(name='Scale', wrap_length = 2190, wrap_size = 365, 
                                 resize = 180, input_size = input_size)(x_hydro)
            lstm   = LSTM(units=16, name='LSTM', 
                                 kernel_regularizer=regularizers.l2(0.001), 
                                 recurrent_regularizer=regularizers.l2(0.001))(x_scale)
            output = Dense(units=2, name='Dense', activation='linear', use_bias=False, 
                                  kernel_regularizer=regularizers.l2(0.001))(lstm)
            model  = Model(x_input, output)
    
        elif model_type == 'physical':
            hydro_output = PRNNLayer(mode='normal', name='Hydro')(x_input)
            model = Model(x_input, hydro_output)
    
        elif model_type == 'lstm':
            x_scale = ScaleLayer(name='Scale', wrap_length = 2190, wrap_size = 365)(x_input)
            lstm   = LSTM(units=16, name='LSTM', 
                                 kernel_regularizer=regularizers.l2(0.001), 
                                 recurrent_regularizer=regularizers.l2(0.001))(x_scale)
            output = Dense(units=2, name='Dense', activation='linear', use_bias=False, 
                                  kernel_regularizer=regularizers.l2(0.001))(lstm)
            model = Model(x_input, output)
    
        return model

    def train_model(model, train_x, train_y, ep_number, lrate, save_path):
        # Train a Keras model.
        # -- model: the Keras model object
        # -- train_x, train_y: the input and target for training the model
        # -- ep_number: the maximum epoch number
        # -- lrate: the initial learning rate
        # -- save_path: where the model will be saved
        
    
        save = callbacks.ModelCheckpoint(save_path, verbose=0, save_best_only=True, 
                                         monitor='nse_metrics', mode='max',
                                         save_weights_only=True)
        es = callbacks.EarlyStopping(monitor='nse_metrics', mode='max', verbose=1, 
                                     patience=20, min_delta=0.005,
                                     restore_best_weights=True)
        reduce = callbacks.ReduceLROnPlateau(monitor='nse_metrics', factor=0.8, patience=5, 
                                             verbose=1, mode='max',
                                             min_delta=0.005, cooldown=0, min_lr=lrate / 100)
        tnan = callbacks.TerminateOnNaN()
    
        model.compile(loss=hydroutils.nse_loss, metrics=[hydroutils.nse_metrics], 
                      optimizer=optimizers.Adam(lr=lrate))
        history = model.fit(train_x, train_y, epochs=ep_number, 
                            batch_size=10000, callbacks=[save, es, reduce, tnan])
    
        return history


    def test_model(model, test_x, save_path):
        # Test a Keras model.
        # -- model: the Keras model object
        # -- test_x: the input for testing the model
        # -- save_path: where the model was be saved
        
        model.load_weights(save_path, by_name=True)
        pred_y = model.predict(test_x, batch_size=10000)
    
        return pred_y

    Path(f'{working_path}/results').mkdir(parents=True, exist_ok=True)
    save_path_hybrid = f'{working_path}/results/{basin_id}_hybrid.h5'
    
    model = create_model((train_x.shape[1], train_x.shape[2]), train_x.shape[0], model_type='hybrid')
    model.summary()
    hybrid_history = train_model(model, train_x, train_y, ep_number=200, lrate=0.01, 
                                  save_path=save_path_hybrid)
    
    Path(f'{working_path}/results').mkdir(parents=True, exist_ok=True)
    save_path_physical = f'{working_path}/results/{basin_id}_physical.h5'
    
    model = create_model((train_x.shape[1], train_x.shape[2]), train_x.shape[0], model_type='physical')
    model.summary()
    hybrid_history = train_model(model, train_x, train_y, ep_number=200, lrate=0.01, 
                                  save_path=save_path_physical)

    def normalize(data):
        data_mean = np.mean(data, axis=-2, keepdims=True)
        data_std = np.std(data, axis=-2, keepdims=True)
        data_scaled = (data - data_mean) / data_std
        return data_scaled, data_mean, data_std

    Path(f'{working_path}/results').mkdir(parents=True, exist_ok=True)
    save_path_common = f'{working_path}/results/{basin_id}_lstm.h5'
    
    model = create_model((train_x.shape[1], train_x.shape[2]), train_x.shape[0], model_type='lstm')
    model.summary()
    
    train_x_nor, train_x_mean, train_x_std = normalize(train_x)
    train_y_nor, train_y_mean, train_y_std = normalize(train_y)
    
    common_history = train_model(model, train_x_nor, train_y_nor, ep_number=200, lrate=0.01, 
                                 save_path=save_path_common)

    ####################
    #  Hybrid DL model #
    ####################
    model = create_model((test_x.shape[1], test_x.shape[2]), model_type='hybrid')
    flow_hybrid = test_model(model, test_x, save_path_hybrid)
    gw_hybrid = flow_hybrid[:, :, 1:]
    flow_water_hybrid = flow_hybrid[:, :, 0:1]


    ####################
    # Physical NN model#
    ####################
    model = create_model((test_x.shape[1], test_x.shape[2]), model_type='physical')
    flow_physical = test_model(model, test_x, save_path_physical)
    gw_physical = flow_physical[:, :, 1:]
    flow_water_physical= flow_physical[:, :, 0:1]
    
    
    ####################
    #  Common NN model #
    ####################
    model = create_model((test_x.shape[1], test_x.shape[2]), model_type='lstm')
    #We use the feature means/stds of the training period for normalization
    test_x_nor = (test_x - train_x_mean) / train_x_std
    
    flow_lstm = test_model(model, test_x_nor, save_path_common)
    #We use the feature means/stds of the training period for recovery
    flow_lstm = flow_lstm * train_y_std + train_y_mean
    gw_lstm= flow_lstm[:, :, 1:]
    flow_water_lstm= flow_lstm[:, :, 0:1]

    evaluate_set = test_set.loc[:, ['prcp(mm/day)','flow(mm)']]
    evaluate_set['flow_obs'] = evaluate_set['flow(mm)']
    evaluate_set['flow_hybrid'] = np.clip(flow_water_hybrid[0, :, :], a_min = 0, a_max = None)
    evaluate_set['flow_physical'] = np.clip(flow_water_physical[0, :, :], a_min = 0, a_max = None)
    evaluate_set['flow_lstm'] = np.clip(flow_water_lstm[0, :, :], a_min = 0, a_max = None)
    
    evaluate_set_2 = test_set.loc[:, ['tmean(C)','GW(feet)']]
    evaluate_set_2['gw_obs'] = evaluate_set_2['GW(feet)']
    evaluate_set_2['gw_hybrid'] = np.clip(gw_hybrid[0, :, :], a_min = 0, a_max = None)
    evaluate_set_2['gw_physical'] = np.clip(gw_physical[0, :, :], a_min = 0, a_max = None)
    evaluate_set_2['gw_lstm'] = np.clip(gw_lstm[0, :, :], a_min = 0, a_max = None)
    
    def addYears(date, years):
        result = date + timedelta(366 * years)
        if years > 0:
            while result.year - date.year > years or date.month < result.month or date.day < result.day:
                result += timedelta(-1)
        elif years < 0:
            while result.year - date.year < years or date.month > result.month or date.day > result.day:
                result += timedelta(1)
        return result
    
    evaluation_start = datetime.strftime(addYears(datetime.strptime(testing_start, '%Y-%m-%d'), 1), '%Y-%m-%d')
    evaluation_end = testing_end
    
    def calc_nse(y_true, y_pred):
        numerator = np.sum((y_pred - y_true) ** 2)
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    
        return 1 - numerator / denominator
    
    # We only evaluate the data in the evaluation period
    date_range = pd.date_range(evaluation_start, evaluation_end)
    evaluate_set = evaluate_set[evaluate_set.index.isin(date_range)]
    
    # Calculate respective NSE values for flow
    nse_hybrid_flow = calc_nse(evaluate_set['flow_obs'].values, evaluate_set['flow_hybrid'].values)
    nse_physical_flow = calc_nse(evaluate_set['flow_obs'].values, evaluate_set['flow_physical'].values)
    nse_LSTM_flow = calc_nse(evaluate_set['flow_obs'].values, evaluate_set['flow_lstm'].values)
    
    evaluate_set_2 = evaluate_set_2[evaluate_set_2.index.isin(date_range)]
    
    # Calculate respective NSE values for GW
    nse_hybrid_gw = calc_nse(evaluate_set_2['gw_obs'].values, evaluate_set_2['gw_hybrid'].values)
    nse_physical_gw = calc_nse(evaluate_set_2['gw_obs'].values, evaluate_set_2['gw_physical'].values)
    nse_LSTM_gw = calc_nse(evaluate_set_2['gw_obs'].values, evaluate_set_2['gw_lstm'].values)


    def evaluation_plot(ax, plot_set, plot_name, line_color, nse_values, basin_id):
        ax.plot(plot_set['flow_obs'], label="observation", color='black', ls='--')
        ax.plot(plot_set[plot_name], label="simulation", color=line_color, lw=1.5)
        ax.set_title(f"Basin {basin_id} - Test set NSE: {nse_values:.3f}")
        ax.set_ylabel("Streamflow (mm/day)")
        ax.legend(loc = 'upper right')
    
    plot_set = evaluate_set[evaluate_set.index.isin(pd.date_range(testing_start, testing_end))]
    
    fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, ncols=1, sharex='row', figsize=(15, 12))
    
    evaluation_plot(ax1, plot_set, 'flow_hybrid', '#e41a1c', nse_hybrid_flow, basin_id)
    evaluation_plot(ax2, plot_set, 'flow_physical', '#377eb8', nse_physical_flow, basin_id)
    evaluation_plot(ax3, plot_set, 'flow_LSTM', '#4daf4a', nse_LSTM_flow, basin_id)
    
    ax1.annotate('Hybrid DL model', xy=(0.05, 0.9), xycoords='axes fraction', size=12)
    ax2.annotate('Physical NN model', xy=(0.05, 0.9), xycoords='axes fraction', size=12)
    ax3.annotate('Common NN model', xy=(0.05, 0.9), xycoords='axes fraction', size=12)
    
    plt.savefig(os.path.join(os.getcwd(),'comparisons','Streamflow for Basin #'+ basin_id + '.png'))

    def evaluation_plot_2(ax, plot_set, plot_name, line_color, nse_values, basin_id):
        ax.plot(plot_set['gw_obs'], label="observation", color='black', ls='--')
        ax.plot(plot_set[plot_name], label="simulation", color=line_color, lw=1.5)
        ax.set_title(f"Basin {basin_id} - Test set NSE: {nse_values:.3f}")
        ax.set_ylabel("GW (feet)")
        ax.legend(loc = 'upper right')
    
    plot_set_2 = evaluate_set_2[evaluate_set_2.index.isin(pd.date_range(testing_start, testing_end))]
    
    fig2, [ax4, ax5, ax6] = plt.subplots(nrows=3, ncols=1, sharex='row', figsize=(15, 12))
    
    evaluation_plot_2(ax4, plot_set_2, 'gw_hybrid', '#e41a1c', nse_hybrid_gw, basin_id)
    evaluation_plot_2(ax5, plot_set_2, 'gw_physical', '#377eb8', nse_physical_gw, basin_id)
    evaluation_plot_2(ax6, plot_set_2, 'gw_LSTM', '#4daf4a', nse_LSTM_gw, basin_id)
    
    ax4.annotate('Hybrid DL model', xy=(0.05, 0.9), xycoords='axes fraction', size=12)
    ax5.annotate('Physical NN model', xy=(0.05, 0.9), xycoords='axes fraction', size=12)
    ax6.annotate('Common NN model', xy=(0.05, 0.9), xycoords='axes fraction', size=12)
    
    plt.savefig(os.path.join(os.getcwd(),'comparisons','GW flow for Basin #'+ basin_id + '.png'))

