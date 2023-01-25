import os
os.system('sleep 10')
quit()

import rdkit
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFMCS, RDKFingerprint, Descriptors, rdMolDescriptors, AllChem, QED
from rdkit import DataStructs


from ddc_pub import ddc_gt

def get_descrs(mols):
    pass
    return descrs_list



target_names = ["logp", "tpsa", "sub_similarity", "qed", "hba", "hbd", "molwt", "gasas"]      
conditions = [4.47190000e+00, 7.75100000e+01, 1.00000000e+00, 3.49746014e-01, 5.00000000e+00, 0.00000000e+00, 4.28256274e+02, 7.00000000e-01]

charset = "Brc1(-23[nH])45C=NOso#FlS67+89%0"
maxlen = 128
name = "ChEMBL25_TRAIN_0813"
dataset_info = {"charset": charset, "maxlen": maxlen, "name": name}

gt = ddc_gt.ddc_gt(target_names=target_names, conditions=conditions)

binmols = [] #从H5文件载入就行
mols = [Chem.Mol(binmol for binmol in binmols)]
descrs = get_descrs(mols)

# 从头训练
##以下init和load二选一
gt.init_model(x=descrs, y=binmols, dataset_info=dataset_info)
gt.load_model(model_name="") #model_name可以是文件名，也可以是DDC的model对象

gt.fit(lr=pow(10,-4.5), lr_init=pow(10,-3))
gt.save_model("models/model_1006")

# 生成分子
## 如果没有load_model就需要load一下（gt.load_model...）
gt.conditions = [4.47190000e+00, 7.75100000e+01, 1.00000000e+00, 3.49746014e-01, 5.00000000e+00, 0.00000000e+00, 4.28256274e+02, 0.7]
gnr_mols = gt.generate(sample_times=250, batch_input_length=128)

# 计算生成分子的描述符
properties = get_descrs(gnr_mols)
gt.set_properties(properties)

# 绘图
gt.plot("results/model_1006.png", title="model_1006")

# 把需要的部分dump成pickle（sani_properties包括rdkit.mol、对应的smiles和所有的描述符信息，二维列表格式）
gt.dump("datasets/generator_1006", gt.sani_properties)

# retrain
## retrain的载入模型和训练是二合一的
filename = "model_1004_retrain_3"
gt.retrain(model_name="models/"+filename, x=descrs, y=binmols, lr=1e-3)