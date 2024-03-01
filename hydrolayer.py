"""
This file is part of the accompanying code to our manuscript:

Jiang S., Zheng Y., & Solomatine D.. (2020) Improving AI system awareness of geoscience knowledge: Symbiotic integration of physical approaches and deep learning. Geophysical Research Letters, 47, e2020GL088229. https://doi.org/10.1029/2020GL088229

Copyright (c) 2020 Shijie Jiang. All rights reserved.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

from keras import initializers, constraints, regularizers
from keras.layers import Layer, LSTM, Dense
import keras.backend as K
import tensorflow as tf
import numpy as np


class PRNNLayer(Layer):
    """Implementation of the standard P-RNN layer
    Hyper-parameters
    ----------
    mode: if in "normal", the output will be the generated flow;
          if in "analysis", the output will be a tensor containing all state variables and process variables
    ==========
    Parameters
    ----------
    f: Rate of decline in flow from catchment bucket | Range: (0, 0.1)
    smax: Maximum storage of the catchment bucket      | Range: (100, 1500)
    qmax: Maximum subsurface flow at full bucket     | Range: (10, 50)
    ddf: Thermal degree‐day factor                     | Range: (0, 5.0)
    tmax: Temperature above which snow starts melting  | Range: (0, 3.0)
    tmin: Temperature below which precipitation is snow| Range: (-3.0, 0)
    """

    def __init__(self, mode='normal', **kwargs):
        self.mode = mode
        super(PRNNLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.f = self.add_weight(name='f', shape=(1,),  #
                                 initializer=initializers.Constant(value=0.5),
                                 constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                 trainable=True)
        self.smax = self.add_weight(name='smax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=1 / 15, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.qmax = self.add_weight(name='qmax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=0.2, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.ddf = self.add_weight(name='ddf', shape=(1,),
                                   initializer=initializers.Constant(value=0.5),
                                   constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                   trainable=True)
        self.tmin = self.add_weight(name='tmin', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.tmax = self.add_weight(name='tmax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.k = self.add_weight(name='k', shape=(1,),
                                 initializer=initializers.Constant(value=0.5),
                                 constraint=constraints.min_max_norm(min_value=0.0, max_value=10, rate=0.9),
                                 trainable=True)

        super(PRNNLayer, self).build(input_shape)

    def heaviside(self, x):
        """
        A smooth approximation of Heaviside step function
            if x < 0: heaviside(x) ~= 0
            if x > 0: heaviside(x) ~= 1
        """

        return (K.tanh(5 * x) + 1) / 2

    def rainsnowpartition(self, p, t, tmin):
        """
        Equations to partition incoming precipitation into rain or snow
            if t < tmin:
                psnow = p
                prain = 0
            else:
                psnow = 0
                prain = p
        """
        tmin = tmin * -3  # (-3.0, 0)

        psnow = self.heaviside(tmin - t) * p
        prain = self.heaviside(t - tmin) * p

        return [psnow, prain]

    def snowbucket(self, s0, t, ddf, tmax):
        """
        Equations for the snow bucket
            if t > tmax:
                if s0 > 0:
                    melt = min(s0, ddf*(t - tmax))
                else:
                    melt = 0
            else:
                melt = 0
        """
        ddf = ddf * 5  # (0, 5.0)
        tmax = tmax * 3  # (0, 3.0)

        melt = self.heaviside(t - tmax) * self.heaviside(s0) * K.minimum(s0, ddf * (t - tmax))

        return melt

    def soilbucket(self, s1, pet, f, smax, qmax, k):
        """
        Equations for the soil bucket
            if s1 < 0:
                et = 0
                qsub = 0
                qsurf = 0
            elif s1 > smax:
                et = pet
                qsub = qmax
                qsurf = s1 - smax
            else:
                et = pet * (s1 / smax)
                qsub = qmax * exp(-f * (smax - s1))
                qsurf = 0
        """
        k = k / 10  # (0, 1)
        f = f / 10  # (0, 0.1)
        smax = smax * 1500  # (100, 1500)
        qmax = qmax * 50  # (10, 50)

        et = self.heaviside(s1) * self.heaviside(s1 - smax) * pet + \
            self.heaviside(s1) * self.heaviside(smax - s1) * pet * (s1 / smax)
        qsub = self.heaviside(s1) * self.heaviside(s1 - smax) * qmax + \
            self.heaviside(s1) * self.heaviside(smax - s1) * qmax * K.exp(-1 * f * (smax - s1))
        qsurf = self.heaviside(s1) * self.heaviside(s1 - smax) * (s1 - smax)
        qst = qsub * k 
        qgr = qsub * (1-k)

        return [et, qsurf, qst, qgr]

    def step_do(self, step_in, states):  # Define step function for the RNN
        s0 = states[0][:, 0:1]  # Snow bucket
        s1 = states[0][:, 1:2]  # Soil bucket

        # Load the current input column
        p = step_in[:, 0:1]
        t = step_in[:, 1:2]
        pet = step_in[:, 2:3]

        # Partition precipitation into rain and snow
        [_ps, _pr] = self.rainsnowpartition(p, t, self.tmin)
        # Snow bucket
        _m = self.snowbucket(s0, t, self.ddf, self.tmax)
        # Soil bucket
        [_et, _qsurf, _qst, _qgr] = self.soilbucket(s1, pet, self.f, self.smax, self.qmax, self.k)

        # Water balance equations
        _ds0 = _ps - _m
        _ds1 = _pr + _m - _et - _qsurf - _qst - _qgr

        # Record all the state variables which rely on the previous step
        next_s0 = s0 + K.clip(_ds0, -1e5, 1e5)
        next_s1 = s1 + K.clip(_ds1, -1e5, 1e5)

        step_out = K.concatenate([next_s0, next_s1], axis=1)

        return step_out, [step_out]

    def call(self, inputs):
        # Load the input vector
        prcp = inputs[:, :, 0:1]
        tmean = inputs[:, :, 1:2]
        dayl = inputs[:, :, 2:3]

        # Calculate PET using Hamon’s formulation
        pet = 29.8 * (dayl * 24) * 0.611 * K.exp(17.3 * tmean / (tmean + 237.3)) / (tmean + 273.2)

        # Concatenate pprcp, tmean, and pet into a new input
        new_inputs = K.concatenate((prcp, tmean, pet), axis=-1)

        # Define 2 initial state variables at the beginning
        init_states = [K.zeros((K.shape(new_inputs)[0], 2))]

        # Recursively calculate state variables by using RNN
        _, outputs, _ = K.rnn(self.step_do, new_inputs, init_states)

        s0 = outputs[:, :, 0:1]
        s1 = outputs[:, :, 1:2]

        # Calculate final process variables
        m = self.snowbucket(s0, tmean, self.ddf, self.tmax)
        [et, qsub, qst, qgr] = self.soilbucket(s1, pet, self.f, self.smax, self.qmax, self.k)

        if self.mode == "normal": 
            return K.concatenate([qsub + qst, qgr], axis=-1)
        elif self.mode == "analysis":
            return K.concatenate([s0, s1, m, et, qsub, qst, qgr], axis=-1)

    def compute_output_shape(self, input_shape):
        if self.mode == "normal":
            return (input_shape[0], input_shape[1], 2)
        elif self.mode == "analysis":
            return (input_shape[0], input_shape[1], 6)


# class LSTMLayer(Layer):
#     """Implementation of the standard 1D-CNN layer

#     Hyper-parameters (same as the Conv1D in https://keras.io/layers/convolutional/)
#     ----------
#     filters: The dimensionality of the output space (i.e. the number of output filters in the convolution)
#     kernel_size: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window
#     padding: One of "valid", "causal" or "same"
#     seed: Random seed for initialization
#     """

#     def __init__(self, units, kernel_regularizer, recurrent_regularizer, **kwargs):
#         self.units = units
#         self.kernel_regularizer = kernel_regularizer
#         self.recurrent_regularizer = recurrent_regularizer
#         super(LSTMLayer, self).__init__(**kwargs)

#     def build(self, input_shape):
#         super(LSTMLayer, self).build(input_shape)

#     def call(self, inputs):
#         outputs = LSTM(units=self.units, 
#                       kernel_regularizer=self.kernel_regularizer, 
#                       recurrent_regularizer=self.recurrent_regularizer)
#         outputs = Dense(units=2,activation='linear', use_bias=False, 
#                           kernel_regularizer=self.kernel_regularizer)
#         return outputs

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], input_shape[1], 2)

class ScaleLayer(Layer):
    """
    Scale the inputs with the mean activation close to 0 and the standard deviation close to 1
    """

    def __init__(self, wrap_length, wrap_size, resize , input_size,**kwargs):
        self.wrap_length = wrap_length
        self.wrap_size = wrap_size
        self.resize = resize
        self.input_size = input_size
        super(ScaleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ScaleLayer, self).build(input_shape)
        
    def data_rewrap(inputs, wrap_length, wrap_size, resize, input_size):
        #unwrap data to rewrap
        for i in range(input_size):
            if i == 0:
                new_df = inputs[i,i:wrap_length,:]
            else:
                new_df += inputs[i, wrap_length + (i-1)*wrap_size:wrap_length + (i)*wrap_size,:]
        #rewrap
        wrap_interval = len(new_df)//resize
        final_df = np.empty(shape=(wrap_size, wrap_length, inputs.shape[2]))
        for i in range(wrap_length):
            final_df[i, :, :] = new_df[resize : (resize + i * wrap_interval), :]
        
        return final_df

    def call(self, inputs):
        wrap_length = self.wrap_length
        wrap_size = self.wrap_size
        resize = self.resize
        input_size = self.input_size
    
        final_df = self.data_rewrap(inputs, wrap_length, wrap_size, resize, input_size)
        
        #Edit this to change inputs to control the features going into LSTM
        met = final_df[:,:, :-2]
        flow = final_df[:,:, -2:-1]
        gw = final_df[:,:, -1:]

        self.met_center = K.mean(met, axis=-2, keepdims=True)
        self.met_scale = K.std(met, axis=-2, keepdims=True)
        self.met_scaled = (met - self.met_center) / self.met_scale
        
        return K.concatenate([self.met_scaled, flow, gw], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape


