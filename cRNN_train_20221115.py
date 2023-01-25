# 提高retrain的epochs
import os, sys
import numpy as np
import random

import rdkit
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFMCS, RDKFingerprint, Descriptors, rdMolDescriptors, AllChem, QED
from rdkit import DataStructs

import gasa

import h5py
import pandas as pd
import ast
import pickle

import ddc_pub
from ddc_pub import ddc_gt

def get_sim(mol, sub_mol) -> float: 
    '''
    计算子结构相似度
    '''
    try:
        res = rdFMCS.FindMCS([mol, sub_mol], timeout=1, bondCompare=rdFMCS.BondCompare.CompareAny, ringMatchesRingOnly=True, atomCompare=rdFMCS.AtomCompare.CompareAny)
        if res.smartsString == "" or res.canceled:
            return 0
        mcs_mol = Chem.MolFromSmarts(res.smartsString)
        Chem.SanitizeMol(mcs_mol)

        mcs_mol_fp = RDKFingerprint(mcs_mol)
        sub_mol_fp = RDKFingerprint(sub_mol)
        sim = DataStructs.FingerprintSimilarity(sub_mol_fp, mcs_mol_fp, metric=DataStructs.DiceSimilarity)

        return sim
    except Exception as e:
        #print(e)
        return 0

def get_gasas(mols: list=[]) -> list:
    '''
    计算GASA打分
    '''
    smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
    gasas = gasa.GASA(smiles_list)[1]
    return gasas

def get_descriptors(mol_list, sub_mol):
    '''
    计算描述符
    顺序："logp", "tpsa", "sub_similarity", "qed", "hba", "hbd", "molwt", "gasas"
    '''
    print("Calculating descriptors for {} mols.".format(len(mol_list)))

    logp_list = [Descriptors.MolLogP(mol) for mol in mol_list]
    tpsa_list = [Descriptors.TPSA(mol) for mol in mol_list]
    sim_list = [get_sim(mol, sub_mol) for mol in mol_list]
    hba_list = [rdMolDescriptors.CalcNumHBA(mol) for mol in mol_list]
    hbd_list = [rdMolDescriptors.CalcNumHBD(mol) for mol in mol_list]
    qed_list = [QED.qed(mol) for mol in mol_list]
    molwt_list = [Descriptors.ExactMolWt(mol) for mol in mol_list]
    gasas_list = get_gasas(mol_list)

    descriptors = [logp_list, tpsa_list, sim_list, qed_list, hba_list, hbd_list, molwt_list, gasas_list]
    return np.asarray(descriptors).T

def dump_data(data, path):
    '''
    导出pickle到指定路径
    '''
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        print("Dumped_" + path)

def read_data(path):
    '''
    读入指定路径的pickle
    '''
    with open(path, 'rb') as f:
        print("Reading_" + path)
        return pickle.load(f)

def filter_data(sani_properties, target_names):
    '''
    筛选sub_similarity>=0.8且按gasa打分排序
    '''
    #print("Saving results.")
    binmols = np.asarray([i[0].ToBinary() for i in sani_properties])
    combined_properties = []
    for idx,binmol in enumerate(binmols):
        combined_properties.append([binmol] + sani_properties[idx][2:])

    combined_data = pd.DataFrame(combined_properties, columns=['binmols']+target_names)
    #print(combined_data.keys())
    combined_data = combined_data.sort_values(['gasas'])
    #print(combined_data.sort_values(['sub_similarity']).tail(10)['sub_similarity'])

    filter_list = [i >= 0.8 for i in combined_data['sub_similarity']]
    '''for idx,sim in enumerate(combined_data['sub_similarity']):
        filter_list.append(float(sim) >= 0.8 and float(combined_data['gasas'][idx]) > 0.25)'''
    filtered_properties_data = combined_data.loc[filter_list]

    filtered_count = len(filtered_properties_data)
    print("Filtered {} mols.".format(filtered_count))

    return (filtered_properties_data, filtered_count)

def get_mols_and_descr(dataset_filename):
    '''
    从已知h5文件中读取分子及其描述符
    '''
    with h5py.File(dataset_filename, "r") as f:
        binmols = np.asarray(f['mols'])

    with h5py.File(dataset_filename[:-3]+"_descrs.h5", "r") as f:
        names = target_names
        descrs = np.concatenate([[np.asarray(f[name]) for name in names]], axis=1)
        descrs = descrs.T

    return (binmols, descrs)

# 公用变量
MODEL_NUMBER = 1115
target_names = ["logp", "tpsa", "sub_similarity", "qed", "hba", "hbd", "molwt", "gasas"]      
sub_smiles = "O=C(OC)C1=CC(C2=O)CCCC2C1=O"
sub_mol = Chem.MolFromSmiles(sub_smiles)
seed_smiles = "CC1(C)C(CCC2(C)C1CCC(C34C)(C)C2CC(C)(C4=O)C(C)=C(C(OC)=O)C3=O)=O"
seed_mol = Chem.MolFromSmiles(seed_smiles)
conditions = [4.47190000e+00, 7.75100000e+01, 1.00000000e+00, 3.49746014e-01, 5.00000000e+00, 0.00000000e+00, 4.28256274e+02, 7.00000000e-01]

charset = "Brc1(-23[nH])45C=NOso#FlS67+89%0"
maxlen = 128
name = "ChEMBL25_TRAIN_{}".format(MODEL_NUMBER)
dataset_info = {"charset": charset, "maxlen": maxlen, "name": name}

gt = ddc_gt.ddc_gt(target_names=target_names, conditions=conditions)

# 使用arg传入运行的步骤、目前运行的次数、其他参数
step = sys.argv[1]
i = eval(sys.argv[2])
if len(sys.argv) >= 4:
    arg = sys.argv[3]

print("Running {} round {}.".format(step, i))

report_names = ['Step', 'Total mols', 'Valid mols without duplicate mols', 'Duplicate mols', 'Validity', 'Filtered mols', 'Filtered rate']
report_list = []

if step == 'generate':
    # 训练集
    print("Loading dataset.")

    # 筛选的相似度0.3-1的分子
    binmols_2, descrs_2 = get_mols_and_descr("../ddc/datasets/CHEMBL25_FILTERED_2.h5")
    # 种子分子
    binmols_3 = np.asarray([seed_mol.ToBinary()] * 3)
    descrs_3 = np.asarray([conditions] * 3)
    # 合并
    binmols_list =  [binmols_3, binmols_2]
    descrs_list = [descrs_3, descrs_2]
    binmols = np.concatenate(binmols_list)
    descrs = np.vstack(descrs_list)

    # 生成分子
    gt.load_model('../ddc/models/model_1111_retrain_4')
    gt.conditions = conditions
    mols, total_count, val_count, dup_count, val = gt.generate(sample_times=10, batch_input_length=128)

    mols_dict = {'mols':mols, 'total_count':total_count, 'val_count':val_count, 'dup_count':dup_count, 'val':val}
    dump_data(mols_dict, '../ddc/datasets/mols_{}_{}.pickle'.format(MODEL_NUMBER,i))

    # 告诉bash脚本生成了多少个分子
    with open('generated_mols_{}'.format(i), 'w') as f:
        f.write(str(val_count))

elif step == 'generate_calc':
    mols_dict = read_data('../ddc/datasets/mols_{}_{}.pickle'.format(MODEL_NUMBER,i))
    
    # 每次根据传入的3位arg切100个出来
    int_arg = int(arg)
    mols_start = 100 * int_arg
    mols_end = 100 * (int_arg + 1)
    mols = mols_dict['mols'][mols_start:mols_end]

    # 计算描述符
    properties = get_descriptors(mols, sub_mol)
    dump_data(properties, '../ddc/datasets/descrs_{}_{}_{}.pickle'.format(MODEL_NUMBER, i, int_arg))

elif step == 'generate_report':
    mols_dict = read_data('../ddc/datasets/mols_{}_{}.pickle'.format(MODEL_NUMBER,i))
    int_arg = int(arg)
    
    # 载入描述符
    properties = read_data('../ddc/datasets/descrs_{}_{}_0.pickle'.format(MODEL_NUMBER, i))
    print('Read {} mol descrs.'.format(len(properties)))
    for idx in range(1, int_arg+1):
        new_properties = read_data('../ddc/datasets/descrs_{}_{}_{}.pickle'.format(MODEL_NUMBER, i, idx))
        print('Read {} mol descrs.'.format(len(properties)))
        properties = np.concatenate([properties, new_properties])
    #properties_data = pd.DataFrame(data=properties, columns=target_names)

    mols_list = [[mol, Chem.MolToSmiles(mol)] for mol in mols_dict['mols']]
    gt.sani_properties = mols_list

    # 设置描述符、绘图
    gt.set_properties(properties)
    gt.plot("results/model_{}_retrain_{}.png".format(MODEL_NUMBER,i), title="model_{}_retrain_{}".format(MODEL_NUMBER,i))

    # 筛选训练集
    filtered_data, filtered_count = filter_data(gt.sani_properties, target_names)

    # 分析生成情况
    val_count = mols_dict['val_count']
    total_count = mols_dict['total_count']
    dup_count = mols_dict['dup_count']
    val = mols_dict['val']

    if val_count == 0:   #防止zero divide
        val_count = 0.1
    report_list = ['Retrain_{}'.format(i), total_count, val_count, dup_count, val, filtered_count, filtered_count/val_count]

    # 导出报告
    report_data = pd.DataFrame(data=[report_list], columns=report_names)
    if i != 0:
        old_data = pd.read_csv('results/report_1115.csv')
        report_data = pd.concat([old_data, report_data])
    report_data.to_csv('results/report_1115.csv')
