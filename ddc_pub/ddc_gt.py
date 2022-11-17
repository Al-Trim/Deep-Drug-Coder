import os, sys
import numpy as np

import pandas as pd
import ast
import pickle

import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import RDKFingerprint
try:
    from ddc_pub import ddc_v3 as ddc
    import ddc_v3 as ddc
except Exception:
    pass

class ddc_gt:
    def __init__(self, target_names:list=[], show_errors:bool=False, conditions:list=[]) -> None:
        '''
        DDC训练器和生成器的结合体, 便于数据互通

        :param target_names: List of target names
        :type target_names: list, optional
        :param show_errors: Whether to show error messages in some cases
        :type show_errors: bool, optional
        :param conditions: 生成条件, 需要生成时再填
        :type conditions: list or np.array, optional
        '''
        self.__data = None
        self.__model = None
        self.__conditions = None
        self.__target_names = target_names
        self.__show_errors = show_errors
        if conditions != []:
            self.__conditions = np.asarray(conditions)
    
    @property
    def target_names(self): '生成条件的标题'; return self.__target_names

    @property
    def sani_mols(self): '生成的分子'; return self.__sani_mols

    @property
    def sani_properties(self): '生成分子的描述符(二维list格式)'; return self.__sani_properties

    @sani_properties.setter
    def sani_properties(self, value):
        self.__sani_properties = value

    @property
    def sani_data(self): '生成数据的DataFrame'; return self.__data
    
    @property
    def model(self): '当前ddc.DDC模型'; return self.__model

    @property
    def conditions(self): '生成条件'; return self.__conditions

    @conditions.setter
    def conditions(self, value):
        self.__conditions = np.asarray(value)

    def set_properties(self, value):
        '''
        设置以生成的分子的性质, 便于绘图等用途
        
        :param value: 二维数组, 要传入的分子性质
        :type value: list
        '''
        assert len(value) == len(self.__sani_properties), "Length not match"
        
        for idx,sani_property in enumerate(self.__sani_properties):
            for column in value[idx]:
                sani_property.append(column)
            #sani_property = sani_property + list(value[idx])
        self.__data = pd.DataFrame(data=self.__sani_properties, columns=['mol', 'smiles']+self.__target_names)

    def init_model(self, x, y, dataset_info=None, scaling=True, noise_std=0.1, lstm_dim=512, dec_layers=3, batch_size=128):
        '''
        新建模型

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
        加载现有模型用于生成

        :param model_name: 路径或ddc.DDC对象
        :type model_name: str or ddc.DDC
        '''
        if type(model_name) == str:
            self.__model = ddc.DDC(model_name = model_name)
        else:
            self.__model = model_name
    
    def save_model(self, filename:str = ""):
        '''
        保存模型

        :param filename: 保存路径
        :type filename: str
        '''
        print("Saving model.")
        self.__model.save(filename)

    def fit(self, model_name:str="New model", 
    epochs=300, lr=1e-3, mini_epochs=10, save_period=50, lr_delay=True, sch_epoch_to_start=500, lr_init=1e-3, lr_final=1e-6, patience=25):
        """
        使用新建模型时输入的数据训练模型
        
        :param model_name: 模型名称, 用于保存训练历史
        :type model_name: str, optional
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

        self.__model.fit(epochs              = epochs,        # number of epochs
        lr                  = lr,                             # initial learning rate for Adam, recommended
        model_name          = model_name,                     # base name to append the checkpoints with
        checkpoint_dir      = "checkpoints_"+model_name+"/",  # save checkpoints in the notebook's directory
        mini_epochs         = mini_epochs,                    # number of sub-epochs within an epoch to trigger lr decay
        save_period         = save_period,                    # checkpoint frequency (in mini_epochs)
        lr_decay            = lr_delay,                       # whether to use exponential lr decay or not
        sch_epoch_to_start  = sch_epoch_to_start,             # mini-epoch to start lr decay (bypassed if lr_decay=False)
        sch_lr_init         = lr_init,                        # initial lr, should be equal to lr (bypassed if lr_decay=False)
        sch_lr_final        = lr_final,                       # final lr before finishing training (bypassed if lr_decay=False)
        patience            = patience)                       # patience for Keras' ReduceLROnPlateau (bypassed if lr_decay=True)

        print("Training completed.")
    
    def retrain(self, model_name, x, y,
    epochs=300, lr=1e-3, mini_epochs=10, save_period=50, lr_delay=True, sch_epoch_to_start=500, lr_init=1e-3, lr_final=1e-6, patience=25):
        """
        从文件载入现有模型并训练之
        
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
        
        self.fit(model_name, epochs, lr, mini_epochs, save_period, lr_delay, sch_epoch_to_start, lr_init, lr_final, patience)
        print("Retrain completed.")
    
    def generate(self, sample_times:int=4, temp:int=1, batch_input_length:int=128):
        """
        根据模型生成多个rdkit mol, 并输出生成的rdkit mol, 分子总数, 有效分子数(去掉重复), 重复分子数, 有效性
        If temp>0, multinomial sampling is used instead of selecting 
        the single most probable character at each step.
        If temp==1, multinomial sampling without temperature scaling is used.
        Low temp leads to elimination of characters with low probabilities.
        
        :param sample_times: Times for sampling
        :type sample_times: int, optional
        :param temp: Temperatute of multinomial sampling (argmax if 0), defaults to 1
        :type temp: int, optional
        :param batch_input_length: The input length of batch, defaults to 128
        :type batch_input_length: int, optional
        :return: tuple(5)
        """
        target = self.__conditions
        print("Sampling with conditions:{:}.".format(target))

        # 从模型中取样、生成
        smiles_out = []
        self.__model.batch_input_length = batch_input_length
        for i in range(sample_times):
            smiles, _ = self.__model.predict_batch(latent=target.reshape(1, -1), temp=temp)
            smiles_out.append(smiles)
        smiles_out = np.concatenate(smiles_out)
        self.__mols = [Chem.MolFromSmiles(smi) for smi in smiles_out]

        # 检查有效性并去重
        print("Checking mols.")

        self.__sani_fps = []
        self.__sani_mols = []
        self.__sani_properties = []

        total_count = len(self.__mols)
        sani_count = 0
        dup_count = 0
        val_count = 0

        for idx,mol in enumerate(self.__mols):
            sani_mol = self.__sanitize(mol)
            if sani_mol != None:
                mol_fp = RDKFingerprint(sani_mol)
                is_dupli = False
                for i in self.__sani_fps:
                    sani_sim = DataStructs.FingerprintSimilarity(mol_fp, i)
                    if sani_sim == 1.0:
                        is_dupli = True
                        dup_count += 1
                        break
                if not is_dupli:
                    self.__sani_mols.append(sani_mol)
                    self.__sani_fps.append(mol_fp)
                    self.__sani_properties.append([sani_mol, smiles_out[idx]])

        sani_count = len(self.__sani_properties)
        val_count = sani_count/total_count

        print("Generated mols:{:}, sanitized mols(include duplicated):{:}, duplicated mols:{:}, validity:{:}".format(
            total_count, sani_count, dup_count, val_count))
        
        return (self.__sani_mols, total_count, sani_count, dup_count, val_count)

    def __sanitize(self, mol):
        '检查分子有效性'
        try:
            Chem.SanitizeMol(mol)
            if Chem.MolToSmiles(mol) == "":
                return None
            else:
                return mol
        except Exception as e:
            if self.__show_errors:
                #print(e)
                pass
            return None
    
    def plot(self, path:str="", title:str="", sharey:bool=False):
        '''
        保存生成分子性质直方图到指定文件

        :param path: 目标文件名
        :type path: str, optional
        :param title: 设置整体标题
        :type title: str, optional
        :param sharey: 控制是否所有子图共享Y轴
        :type sharey: bool, optional
        '''
        assert self.__data is not None, "Set sani_properties first before plotting"
        assert self.__conditions is not None, "Set conditions first before plotting"
        
        target_length = len(self.__target_names)
        fig, axs = plt.subplots(nrows=1, ncols=target_length, sharey=sharey, figsize=(target_length*2, 2))
        for idx,target_name in enumerate(self.__target_names):
            cdt = self.__conditions[idx]
            subplot = axs[idx]

            ax_cdt = subplot.twiny()
            ax_cdt.hist(self.__data[target_name], 25, color='#F5B3B3')
            ax_cdt.set_xticks([cdt])

            subplot.hist(self.__data[target_name], 25, color='w')
            subplot.set_title(target_name)
            ax_cdt.axvline(x=cdt, color='r', linestyle='dashed')

            #ax_cdt.set_xticklabels([str(cdt)])
            #subplot.text(x=cdt, y=0, s=str(cdt), color='r')
        
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path)
        print("Figure saved to \"{:}\".".format(path))

    def dump(self, filename:str = "", data = None):
        '''
        使用pickle导出对象

        :param filename: 目标文件名
        :type filename: str
        :param data: 要导出的对象
        :type data: Any
        '''
        with open(filename, "wb") as f:
            pickle.dump(data, f)
    
    def load_data_old(self, mols_filename:str = "", properties_filename:str = ""):
        '''
        加载0924前导出的生成数据

        :param mols_filename: 分子对应的pickle文件
        :type mols_filename: str
        :param properties_filename: 描述符对应的pickle文件
        :type properties_filename: str
        '''
        with open(mols_filename, "rb") as f:
            sani_mols = pickle.load(f)
        with open(properties_filename, "rb") as f:
            sani_properties = pickle.load(f)

        assert len(sani_mols) == len(sani_properties)
        for i in range(len(sani_mols)):
            sani_properties[i].insert(0, '')
            sani_properties[i].insert(0, sani_mols[i])

        self.__sani_properties = sani_properties
        self.__data = pd.DataFrame(self.__sani_properties, columns=['mol','smiles']+self.__target_names)
        print("Loaded {} mols with properties.".format(len(self.__data))) 
    
    def load_data(self, properties_filename:str = ""):
        '''
        加载使用pickle导出的sani_properties

        :param properties_filename: 要加载的pickle文件
        :type properties_filename: str
        '''
        with open(properties_filename, "rb") as f:
            self.__sani_properties = pickle.load(f)
        self.__data = pd.DataFrame(self.__sani_properties, columns=['mol','smiles']+self.__target_names) 
        print("Loaded {} mols with properties.".format(len(self.__data))) 

