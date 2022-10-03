import os, sys
import numpy as np

import pandas as pd
import ast
import pickle

import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs, RDKFingerprint
import ddc_v3 as ddc

class ddc_gt:
    def __init__(self, target_names:list=[], show_errors:bool=False) -> None:
        '''
        The combination of DDC trainer and generator, providing with an easier way to use DDC

        :param target_names: List of target names
        :type target_names: list, optional
        :param show_errors: Whether to show error messages in some cases
        :type show_errors: bool, optional
        '''
        self.__data = None
        self.__conditions = None
        self.__target_names = target_names
        self.__show_errors = show_errors
    
    @property
    def target_names(self): 'List of target names'; return self.__target_names

    @property
    def sani_properties(self): 'Properties of generated mols'; return self.__sani_properties

    @property.setter
    def sani_properties(self, value:list=[]):
        '''
        Set the properties of generated mols
        Note: this will not replace the whole properties list, but will append the values passed
        '''
        assert len(value) == len(self.__sani_properties), "Length not match"
        for idx,property in enumerate(self.__sani_properties):
            property.append(value[idx])
        self.__data = pd.DataFrame(data=self.__sani_properties, columns=['mol', 'smiles']+self.__target_names)

    @property
    def sani_data(self): 'Generated mols with descrs'; return self.__data
    
    @property
    def model(self): 'The DDC model'; return self.__model

    @property
    def conditions(self): 'Conditions for generating'; return self.__conditions

    @property.setter
    def conditions(self, value):
        'Set the conditions for generating'
        self.__conditions = np.asarray(value)

    def init_model(self, x, y, dataset_info=None, scaling=True, noise_std=0.1, lstm_dim=512, dec_layers=3, batch_size=128):
        '''
        Initialize a new ddc model

        :param x: Encoder input
        :type x: list or numpy.ndarray
        :param y: Decoder input for teacher's forcing
        :type y: list or numpy.ndarray
        :param dataset_info: Metadata about dataset
        :type dataset_info: dict
        :param scaling: Flag to scale descriptor inputs, defaults to `False`
        :type scaling: boolean
        :param noise_std: Standard deviation of noise in the latent space, defaults to 0.01
        :type noise_std: float
        :param lstm_dim: Number of LSTM units in the encoder/decoder layers, defaults to 256
        :param dec_layers: Number of decoder layers, defaults to 2
        :type dec_layers: int
        :param batch_size: Batch size to train with, defaults to 256
        :type batch_size: int
        '''
        if dataset_info is None:
            print("Dataset info not set.")
            return
        self.__model = ddc.DDC(
            x              = x,
            y              = y,
            dataset_info   = dataset_info, 
            scaling        = scaling,
            noise_std      = noise_std,
            lstm_dim       = lstm_dim,
            dec_layers     = dec_layers,
            batch_size     = batch_size
            )
        print("New model initialized.")
    
    def load_model(self, model_name):
        '''
        Load existing model for generating

        :param model_name: The path of existing model or the DDC model itself
        :type model_name: str or ddc.DDC
        '''
        
        if type(model_name) == str:
            self.__model = ddc.DDC(model_name = model_name)
        else:
            self.__model = model_name
    
    def fit(self, epochs=300, lr=1e-3, mini_epochs=10, save_period=50, lr_delay=True, sch_epoch_to_start=500, lr_init=1e-3, lr_final=1e-6, patience=25):
        """
        Fit the full model to the training data.
        
        :param epochs: Training iterations over complete training set.
        :type epochs: int
        :param lr: Initial learning rate
        :type lr: float
        :param mini_epochs: Subdivisions of a single epoch to trick Keras into applying callbacks
        :type mini_epochs: int
        :param save_period: Checkpoint period in miniepochs, defaults to 5
        :type save_period: int, optional
        :param lr_decay: Flag to enable exponential learning rate decay, defaults to False
        :type lr_decay: bool, optional
        :param sch_epoch_to_start: Miniepoch to start exponential learning rate decay, defaults to 500
        :type sch_epoch_to_start: int, optional
        :param lr_init: Initial learning rate to start exponential learning rate decay, defaults to 1e-3
        :type lr_init: float, optional
        :param lr_final: Target learning rate value to stop decaying, defaults to 1e-6
        :type lr_final: float, optional
        :param patience: minimum consecutive mini_epochs of stagnated learning rate to consider before lowering it with ReduceLROnPlateau 
        :type patience: int
        """
        assert type(self.__model) == ddc.DDC, "Model to fit uninitialized."
        print("Training model.")

        self.__model.fit(epochs              = epochs,                               # number of epochs
                        lr                  = lr,                                  # initial learning rate for Adam, recommended
                        model_name          = self.model_name,                     # base name to append the checkpoints with
                        checkpoint_dir      = "checkpoints_"+self.model_name+"/",  # save checkpoints in the notebook's directory
                        mini_epochs         = mini_epochs,                         # number of sub-epochs within an epoch to trigger lr decay
                        save_period         = save_period,                         # checkpoint frequency (in mini_epochs)
                        lr_decay            = lr_delay,                            # whether to use exponential lr decay or not
                        sch_epoch_to_start  = sch_epoch_to_start,                  # mini-epoch to start lr decay (bypassed if lr_decay=False)
                        sch_lr_init         = lr_init,                             # initial lr, should be equal to lr (bypassed if lr_decay=False)
                        sch_lr_final        = lr_final,                            # final lr before finishing training (bypassed if lr_decay=False)
                        patience            = patience)                            # patience for Keras' ReduceLROnPlateau (bypassed if lr_decay=True)

        print("Training completed.")
    
    def retrain(self, model_name, x, y,
    epochs=300, lr=1e-3, mini_epochs=10, save_period=50, lr_delay=True, sch_epoch_to_start=500, lr_init=1e-3, lr_final=1e-6, patience=25):
        """
        Load an existing DDC model and fit the full model to the training data.
        
        :param model_name: The path of existing model (not DDC model)
        :type model_name: str
        :param x: Encoder input
        :type x: list or numpy.ndarray
        :param y: Decoder input for teacher's forcing
        :type y: list or numpy.ndarray
        :param epochs: Training iterations over complete training set.
        :type epochs: int
        :param lr: Initial learning rate
        :type lr: float
        :param mini_epochs: Subdivisions of a single epoch to trick Keras into applying callbacks
        :type mini_epochs: int
        :param save_period: Checkpoint period in miniepochs, defaults to 5
        :type save_period: int, optional
        :param lr_decay: Flag to enable exponential learning rate decay, defaults to False
        :type lr_decay: bool, optional
        :param sch_epoch_to_start: Miniepoch to start exponential learning rate decay, defaults to 500
        :type sch_epoch_to_start: int, optional
        :param lr_init: Initial learning rate to start exponential learning rate decay, defaults to 1e-3
        :type lr_init: float, optional
        :param lr_final: Target learning rate value to stop decaying, defaults to 1e-6
        :type lr_final: float, optional
        :param patience: minimum consecutive mini_epochs of stagnated learning rate to consider before lowering it with ReduceLROnPlateau 
        :type patience: int
        """
        self.__model = ddc.DDC(x=x, y=y, model_name=model_name)
        
        self.fit(epochs, lr, mini_epochs, save_period, lr_delay, sch_epoch_to_start, lr_init, lr_final, patience)
        print("Retrain completed.")
    
    def generate(self, sample_times:int=4, temp:int=1, conditions:list=[], batch_input_length:int=128):
        """Generate multiple biased SMILES strings.
        If temp>0, multinomial sampling is used instead of selecting 
        the single most probable character at each step.
        If temp==1, multinomial sampling without temperature scaling is used.
        Low temp leads to elimination of characters with low probabilities.
        
        :param sample_times: Times for sampling
        :type sample_times: int, optional
        :param temp: Temperatute of multinomial sampling (argmax if 0), defaults to 1
        :type temp: int, optional
        :param conditions: Specific conditions for generating
        :type conditions: list
        :param batch_input_length: The input length of batch, defaults to 128
        :type batch_input_length: int, optional
        """
        if conditions != []:
            self.conditions = conditions 
        target = self.__conditions
        print("Sampling with conditions:{:}.".format(target))

        # Sample
        smiles_out = []
        self.__model.batch_input_length = batch_input_length
        for i in range(sample_times):
            smiles, _ = self.__model.predict_batch(latent=target.reshape(1, -1), temp=temp)
            #print("#{:}:{:}".format(i,smiles))
            smiles_out.append(smiles)
        smiles_out = np.concatenate(smiles_out)
        self.__mols = [Chem.MolFromSmiles(smi) for smi in smiles_out]

        # Sanitize
        print("Checking mols.")
        self.__sani_fps = []
        self.__sani_properties = []
        for idx,mol in enumerate(self.__mols):
            sani_mol = self.__sanitize(mol)
            if sani_mol != None:
                #去重
                mol_fp = RDKFingerprint(sani_mol)
                is_dupli = False
                for i in self.__sani_fps:
                    sani_sim = DataStructs.FingerprintSimilarity(mol_fp, i)
                    if sani_sim == 1.0:
                        is_dupli = True
                        break
                if not is_dupli:
                    self.__sani_fps.append(mol_fp)
                    self.__sani_properties.append(sani_mol, smiles_out[idx])
        print("Generated mols:{:}, sanitized mols:{:}, validity:{:}".format(
            len(self.__mols), len(self.__sani_properties), len(self.__sani_properties)/len(self.__mols)))

    def __sanitize(self, mol):
        'To check whether the mol is invalid'
        try:
            Chem.SanitizeMol(mol)
            if Chem.MolToSmiles(mol) == "":
                return None
            else:
                return mol
        except Exception as e:
            if self.__show_errors:
                print(e)
            return None
    
    def plot(self, path:str=""):
        '''
        Save the histogram of properties of generated results to file

        :param path: Path to save the image
        :type path: str, optional
        '''
        assert self.__data is not None, "Set sani_properties first before plotting"
        assert self.__conditions is not None, "Set conditions first before plotting"
        
        fig, axs = plt.subplots(nrows=3, ncols=3, sharey=True, figsize=(8,6))
        for idx,column in enumerate(self.__data[2:]):
            axs[idx].hist(column, 25)
            axs[idx].set_title(self.__target_names[idx])
            axs[idx][0].vlines(self.__conditions[idx], colors='g', linestyles='dashed', linewidth=2)
        
        fig.savefig(path)
        print("Figure saved to \"{:}\".".format(path))
