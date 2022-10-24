import pandas as pd
import numpy as np
import seaborn as sns
import powerlaw as pl
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit_heatmaps.molmapping import mapvalues2mol
from rdkit_heatmaps.utils import transform2png
import matplotlib.pylab as plt
import os
from collections import defaultdict

def make_ss_dict(dir, type='drug'):
    num_dict = dict()
    sal_dict = dict()
    for filename in os.listdir(dir):
        if type == 'drug':
            name = filename.split('_')[1]
        else:
            name = filename.split('_')[2]

        one = np.load(os.path.join(dir, filename))
        if name not in num_dict.keys() and name not in sal_dict.keys():
            num_dict[name] = 1
            sal_dict[name] = one
        else:
            num_dict[name] += 1
            sal_dict[name] = np.add(sal_dict[name], one)
    
    for k, v in sal_dict.items():
        sal_dict[k] = v/num_dict[k]
    
    return num_dict, sal_dict


# def make_cell_dict(dir):
#     drug_dict = dict()
#     sal_dict = dict()
#     for filename in os.listdir(dir):
#         cell_name = filename.split('_')[2]
#         one = np.load(os.path.join(dir, filename))
#         if cell_name not in drug_dict.keys() and cell_name not in sal_dict.keys():
#             drug_dict[cell_name] = 1
#             sal_dict[cell_name] = one
#         else:
#             drug_dict[cell_name] += 1
#             sal_dict[cell_name] = np.add(sal_dict[cell_name], one)
    
#     for k, v in sal_dict.items():
#         sal_dict[k] = v/drug_dict[k]
    
#     return drug_dict, sal_dict


def make_edge_dict(loader):
    smiles_dict = dict()
    edge_index_dict = dict()
    for data in loader:
        drug_name = data.drug_name[0]
        smiles = data.smiles[0]
        edge_index = data.edge_index.numpy()
        
        if drug_name not in smiles_dict.keys() and drug_name not in edge_index_dict.keys():
            smiles_dict[drug_name] = smiles
            edge_index_dict[drug_name] = edge_index
    
    return smiles_dict, edge_index_dict


def draw_mol_saliency_scores(drug_sal_dict, smiles_dict, edge_index_dict, save_path, annotation_type):
    for k, v in drug_sal_dict.items():
        print('working on ', k)
        edge_index = edge_index_dict[k]
        # norm_v = (v - v.min()) /(v.max() - v.min())

        edge_ss_dict = defaultdict(float)
        counts = defaultdict(int)

        # for val, x, y in zip(norm_v, *edge_index):
        for val, x, y in zip(v, *edge_index):
            if x > y:
                x, y = y, x
            edge_ss_dict[(x, y)] += val
            counts[(x, y)] += 1
            
        for edge, count in counts.items():
            edge_ss_dict[edge] /= count
            # edge_ss_dict[edge] = edge_ss_dict[edge].round(2)
            
        min_ss = min(edge_ss_dict.values())
        max_ss = max(edge_ss_dict.values())
        # print(min_ss, max_ss)

        for edge, value in edge_ss_dict.items():
            edge_ss_dict[edge] = (value - min_ss)/(max_ss - min_ss)
            edge_ss_dict[edge] = edge_ss_dict[edge].round(2)

        smiles = smiles_dict[k]
        mol = Chem.MolFromSmiles(smiles)
        AllChem.EmbedMolecule(mol)
        AllChem.Compute2DCoords(mol)
        bond_weights = []
        for i, bond in enumerate(mol.GetBonds()):
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if u > v:
                u, v = v, u
            if annotation_type == 0:
                bond.SetProp('bondNote',str(edge_ss_dict[(u, v)]))
            elif annotation_type == 1:
                bond_weights.append(edge_ss_dict[(u, v)])
            elif annotation_type == 2:
                bond.SetProp('bondNote',str(edge_ss_dict[(u, v)]))
                bond_weights.append(edge_ss_dict[(u, v)])
        
        if annotation_type == 0:
            Chem.Draw.MolToImageFile(mol, os.path.join(save_path, k + '.png'), size = (1000, 1000))
        elif annotation_type == 1 or annotation_type == 2:
            canvas = mapvalues2mol(mol, bond_weights = bond_weights)
            img = transform2png(canvas.GetDrawingText())
            img.save(os.path.join(save_path, k + '.png'))


def normalize_ss(sal_dict):
    for k, v in sal_dict.items():
        max_ss = np.max(v)
        min_ss = np.min(v)
        sal_dict[k] = (v - min_ss) / (max_ss - min_ss)
    
    return sal_dict


def rank_ss(sal_dict):
    rank_dict = dict()
    for k, v in sal_dict.items():
        temp = np.argsort(-1*(v.reshape(-1)))
        rank_dict[k] = temp
        
    return rank_dict


def one_shot_removal(feature_score, alpha):
    """
    Fits the distribution of saliency score to various distributions, find the best fitting one and keep alpha % of the features
    Performed for a single layer; this function is called by called by compute_new_reduced_model 
    Inputs:
    - feature_score: Numpy array containing the saliency score for each feature
    - alpha: 1 - alpha represents the fraction of (the most important) features to keep (float)
    Returns: 
    - selected_features: Numpy array containing 1s and 0s, 1 represents a selected feature 
    """

    selected_features = np.zeros(np.shape(feature_score))

    LAYER_SIZE_THRESHOLD = 2 

    if np.shape(feature_score)[0] > LAYER_SIZE_THRESHOLD:
        feature_score[feature_score == 0] = 1e-10
        x_min = np.min(feature_score) 
        x_max = np.max(feature_score) 
        params_power_law, loglikelihood_power_law = pl.distribution_fit(np.asarray(feature_score), distribution='power_law', xmin=x_min, xmax=x_max, discrete=False, comparison_alpha=False, search_method='Likelihood', estimate_discrete=False)
        params_lognormal, loglikelihood_lognormal = pl.distribution_fit(np.asarray(feature_score), distribution='lognormal', xmin=x_min, xmax=x_max, discrete=False, comparison_alpha=False, search_method='Likelihood', estimate_discrete=False)
        params_expo, loglikelihood_expo = pl.distribution_fit(np.asarray(feature_score), distribution='exponential', xmin=x_min, xmax=x_max, discrete=False, comparison_alpha=False, search_method='Likelihood', estimate_discrete=False)
        params_stretched, loglikelihood_stretched = pl.distribution_fit(np.asarray(feature_score), distribution='stretched_exponential', xmin=x_min, xmax=x_max, discrete=False, comparison_alpha=False, search_method='Likelihood', estimate_discrete=False)

        print('Shape of layer', np.shape(feature_score))
        print('loglikelihood_power_law', loglikelihood_power_law, 'loglikelihood_lognormal', loglikelihood_lognormal, 'loglikelihood_expo', loglikelihood_expo, 'loglikelihood_stretched', loglikelihood_stretched) 

        if loglikelihood_power_law > max(loglikelihood_lognormal, loglikelihood_expo, loglikelihood_stretched):  
            theoretical_distribution = pl.Power_Law(xmin=x_min, parameters=params_power_law, xmax=x_max, discrete=False)
            prob_dist = theoretical_distribution.cdf(feature_score)
            best_fit_dist = 'Power_Law'
            best_param = params_power_law

        elif loglikelihood_lognormal > max(loglikelihood_power_law, loglikelihood_expo, loglikelihood_stretched):
            theoretical_distribution = pl.Lognormal(xmin=x_min, parameters=params_lognormal, xmax=x_max, discrete=False)
            prob_dist = theoretical_distribution.cdf(feature_score)
            best_fit_dist = 'Lognormal'
            best_param = params_lognormal

        elif loglikelihood_expo > max(loglikelihood_power_law, loglikelihood_lognormal, loglikelihood_stretched):
            theoretical_distribution = pl.Exponential(xmin=x_min, parameters=params_expo, xmax=x_max, discrete=False)
            prob_dist = theoretical_distribution.cdf(feature_score)
            best_fit_dist = 'Exponential'
            best_param = params_expo

        elif loglikelihood_stretched > max(loglikelihood_power_law, loglikelihood_lognormal, loglikelihood_expo):
            theoretical_distribution = pl.Stretched_Exponential(xmin=x_min, parameters=params_stretched, xmax=x_max, discrete=False)
            prob_dist = theoretical_distribution.cdf(feature_score)
            best_fit_dist = 'Stretched_Exponential'
            best_param = params_stretched

        print('values', feature_score)
        print('PDF: ', prob_dist, prob_dist.shape, 'best fit distribution', best_fit_dist, 'best params ', best_param)
        selected_features = prob_dist > (1 - alpha)
        
        print('Number of DeepLIFT selected features: ', np.sum(selected_features))

    if np.shape(feature_score)[0] < LAYER_SIZE_THRESHOLD or np.sum(selected_features) == 0:
        selected_features = np.ones(np.shape(feature_score))

    return selected_features


def draw_one(save_path, name, ranked_ss, ranked_genes, top_n=25):
    values = ranked_ss[:top_n]
    index = ranked_genes[:top_n]
    
    plt.clf()
    fig, ax = plt.subplots(figsize=(250, 20))
    sns.set(font_scale=5)


    res = sns.heatmap(values.reshape(1, -1))
    # plt.xticks(np.arange(100), temp[:100])
    res.set_xticklabels(index)
    res.set_yticklabels([])

    # plt.show()
    fig.savefig(os.path.join(save_path, name + '.png'))


def draw_gene_saliency(rank_dict, sal_dict, gene_list, save_path):
    i = 0
    for key in sal_dict.keys():
        i += 1
        print('working on ', key)
        print('progress: ', i, '/', len(sal_dict))
        rnk = rank_dict[key]
        sal_score = sal_dict[key].reshape(-1)
        ranked_ss = sal_score[rnk]
        ranked_genes = gene_list[rnk]

        draw_one(save_path, key, ranked_ss, ranked_genes)