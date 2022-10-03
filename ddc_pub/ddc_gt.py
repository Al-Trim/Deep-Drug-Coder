import os, sys
from re import X
import numpy as np
import random

import h5py
import pandas as pd
import ast
import pickle

from ddc_pub import ddc_v3 as ddc

class ddc_gt:
    def __init__(self, x, y) -> None:
        '''
        The combination of ddc trainer and generator
        '''
        self.__data = None
        self.__x = x
        self.__y = y
    
    @property
    def generated_data(self):
        return self.__data
    
    def init_model(self, dataset_info=None, scaling=True, noise_std=0.1, lstm_dim=512, dec_layers=3, batch_size = 128):
        '''
        Initialize a new ddc model
        
        '''
        if dataset_info is None:
            print("Dataset info not set.")
            return

        self.model = ddc.DDC(
            x              = self.__x,     # input
            y              = self.__y,     # output
            dataset_info   = dataset_info, # dataset information
            scaling        = scaling,      # scale the descriptors
            noise_std      = noise_std,    # std of the noise layer
            lstm_dim       = lstm_dim,     # breadth of LSTM layers
            dec_layers     = dec_layers,   # number of decoding layers
            batch_size     = batch_size    # batch size for training
            )
        print("New model initialized.")

