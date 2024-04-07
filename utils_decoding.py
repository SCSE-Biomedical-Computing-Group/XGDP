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
from tqdm import tqdm
from collections import defaultdict
from IPython.display import SVG, Image
from rdkit import Chem
from rdkit import Geometry
from rdkit.Chem import rdDepictor,Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import re
from EFGs import mol2frag
import cairosvg

def make_ss_dict(atom_dir, bond_dir, type='drug'):
    num_dict = dict()
    atom_sal_dict = dict()
    bond_sal_dict = dict()
    for filename in os.listdir(bond_dir):
        if type == 'drug':
            name = filename.split('_')[1]
        else:
            name = filename.split('_')[2].split('.')[0]

        # print(filename)
        one_atom = np.load(os.path.join(atom_dir, filename))    # filename is the same for atom and bond
        one_atom = one_atom.reshape(-1)
        one_bond = np.load(os.path.join(bond_dir, filename))
        if name not in num_dict.keys() and name not in bond_sal_dict.keys():
            num_dict[name] = 1
            atom_sal_dict[name] = one_atom
            bond_sal_dict[name] = one_bond
        else:
            num_dict[name] += 1
            atom_sal_dict[name] = np.add(atom_sal_dict[name], one_atom)
            bond_sal_dict[name] = np.add(bond_sal_dict[name], one_bond)
    
    for k, v in bond_sal_dict.items():
        bond_sal_dict[k] = v/num_dict[k]

    for k, v in atom_sal_dict.items():
        atom_sal_dict[k] = v/num_dict[k]
    
    return num_dict, atom_sal_dict, bond_sal_dict

def make_gene_ss_dict(dir, type='drug'):
    num_dict = dict()
    sal_dict = dict()
    for filename in os.listdir(dir):
        if type == 'drug':
            name = filename.split('_')[1]
        else:
            name = filename.split('_')[2].split('.')[0]

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


def draw_mol_saliency_scores(node_sal_dict, edge_sal_dict, smiles_dict, edge_index_dict, save_path, annotation_type):
    '''
        use the rdkit_heatmap pkg to visualize the saliency scores
    '''
    for k, v in tqdm(edge_sal_dict.items()):
        # print('working on ', k)

        # TODO: write standardize function of bond and atom saliency scores
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
            # edge_ss_dict[edge] = (value - min_ss)/(max_ss - min_ss)
            edge_ss_dict[edge] = -1 + 2*(value - min_ss)/(max_ss - min_ss)
            edge_ss_dict[edge] = edge_ss_dict[edge].round(2)

        stand_node_sal = -1 + 2*(node_sal_dict[k] - node_sal_dict[k].min())/(node_sal_dict[k].max() - node_sal_dict[k].min())
        stand_node_sal = stand_node_sal.round(2)

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
            else:
                raise ValueError('annotation_type should be 0, 1, or 2')
        
        for i, atom in enumerate(mol.GetAtoms()):
            if annotation_type == 0:
                atom.SetProp('atomNote',str(stand_node_sal[i]))
            elif annotation_type == 1:
                pass
            elif annotation_type == 2:
                atom.SetProp('atomNote',str(stand_node_sal[i]))
            else:
                raise ValueError('annotation_type should be 0, 1, or 2')
            
        if annotation_type == 0:
            Chem.Draw.MolToImageFile(mol, os.path.join(save_path, k + '.png'), size = (1000, 1000))
        elif annotation_type == 1 or annotation_type == 2:
            canvas = mapvalues2mol(mol, atom_weights = stand_node_sal, bond_weights = bond_weights, color='bwr', value_lims=[-1,1])
            img = transform2png(canvas.GetDrawingText())
            img.save(os.path.join(save_path, k + '.png'))


class drug_sal:
    def __init__(self, name, smiles, node_score, edge_score, edge_idx):
        self.name = name
        self.smiles = smiles
        self.node_score = node_score
        self.edge_score = edge_score
        self.edge_idx = edge_idx
        
    def decomp_fg(self, decoding_voc):
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.fg, self.non_fg, self.fg_idx, self.non_fg_idx = mol2frag(self.mol, toEnd=True, vocabulary=list(decoding_voc), returnidx=True)
        self.fg_atom_idx = [idx for subtuple in self.fg_idx for idx in subtuple]
        self.non_fg_atom_idx = [idx for subtuple in self.non_fg_idx for idx in subtuple]
        self.group_atom_idx = []
        self.single_atom_idx = []
        for subtuple in self.fg_idx+self.non_fg_idx:
            if len(subtuple) == 1:
                self.single_atom_idx.append(subtuple[0])
            else:
                self.group_atom_idx.append([idx for idx in subtuple])

    def decomp_atom(self):
        self.mol = Chem.MolFromSmiles(self.smiles)
        
    def compute_sal_score(self):
        # node:
        # stand_node_sal = -1 + 2*(self.node_score - self.node_score.min())/(self.node_score.max() - self.node_score.min())
        stand_node_sal = (self.node_score - self.node_score.min())/(self.node_score.max() - self.node_score.min())
        stand_node_sal = stand_node_sal.round(2)
        new_sal_dict = {i:stand_node_sal[i] for i in self.single_atom_idx}
        
        for idxes in self.group_atom_idx:
            num = len(idxes)
            fg_sal = [stand_node_sal[i] for i in idxes]
            fg_score = sum(fg_sal)/num
            new_sal_dict[idxes[0]] = fg_score
            
        self.node_sal_dict = new_sal_dict    # this is a dict
        
        # edge:
        edge_ss_dict = defaultdict(float)
        counts = defaultdict(int)
        for val, x, y in zip(self.edge_score, *self.edge_idx):
            if x > y:
                x, y = y, x
            edge_ss_dict[(x, y)] += val
            counts[(x, y)] += 1
            
        for edge, count in counts.items():
            edge_ss_dict[edge] /= count
            
        min_ss = min(edge_ss_dict.values())
        max_ss = max(edge_ss_dict.values())
        
        for edge, value in edge_ss_dict.items():
            edge_ss_dict[edge] = (value - min_ss)/(max_ss - min_ss)
            # edge_ss_dict[edge] = -1 + 2*(value - min_ss)/(max_ss - min_ss)
            edge_ss_dict[edge] = edge_ss_dict[edge].round(2)
            
        bond_weights = []
        for i, bond in enumerate(self.mol.GetBonds()):
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if u > v:
                u, v = v, u
            bond_weights.append(edge_ss_dict[(u, v)])
            
        self.edge_sal = bond_weights    # this is a list

    def compute_sal_score_atom_level(self):
        # node:
        stand_node_sal = (self.node_score - self.node_score.min())/(self.node_score.max() - self.node_score.min())
        self.node_sal_dict = stand_node_sal.round(2)
        
        # edge:
        edge_ss_dict = defaultdict(float)
        counts = defaultdict(int)
        for val, x, y in zip(self.edge_score, *self.edge_idx):
            if x > y:
                x, y = y, x
            edge_ss_dict[(x, y)] += val
            counts[(x, y)] += 1
            
        for edge, count in counts.items():
            edge_ss_dict[edge] /= count
            
        min_ss = min(edge_ss_dict.values())
        max_ss = max(edge_ss_dict.values())
        
        for edge, value in edge_ss_dict.items():
            edge_ss_dict[edge] = (value - min_ss)/(max_ss - min_ss)
            # edge_ss_dict[edge] = -1 + 2*(value - min_ss)/(max_ss - min_ss)
            edge_ss_dict[edge] = edge_ss_dict[edge].round(2)
            
        bond_weights = []
        for i, bond in enumerate(self.mol.GetBonds()):
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if u > v:
                u, v = v, u
            bond_weights.append(edge_ss_dict[(u, v)])
            
        self.edge_sal = bond_weights    # this is a list

    def in_same_fg(self, atom1, atom2):
        for group in self.group_atom_idx:
            if atom1 in group and atom2 in group:
                return True
        return False
        
    def compute_color(self):
        my_cmap = cm.get_cmap('coolwarm')
        patt = r'[C,H][0-9]{2}[0,-1,1]'
        # Get normalize function (takes data in range [vmin, vmax] -> [0, 1])
        my_norm = Normalize(vmin=0, vmax=1)
        
        self.atommap, self.bondmap = {}, {}
        for s,i in zip(self.fg+self.non_fg, self.fg_idx+self.non_fg_idx):
            # print("s and i:", s, i)
            self.atommap.update({x:my_cmap(my_norm(self.node_sal_dict[i[0]]))[:3] for x in i})
            
        for b in self.mol.GetBonds():
            b_id = b.GetIdx()
            score = self.edge_sal[b_id]
            if self.in_same_fg(b.GetBeginAtomIdx(), b.GetEndAtomIdx()):
                self.bondmap[b_id] = self.atommap[b.GetBeginAtomIdx()]
            
        self.highlights = {
            "highlightAtoms": list(self.atommap.keys()),
            "highlightAtomColors": self.atommap,
            "highlightBonds": list(self.bondmap.keys()),
            "highlightBondColors": self.bondmap,
        }

    def compute_color_atom_level(self):
        my_cmap = cm.get_cmap('coolwarm')
        my_norm = Normalize(vmin=0, vmax=1)

        self.atommap = {i:my_cmap(my_norm(self.node_sal_dict[i]))[:3] for i in range(len(self.node_sal_dict))}
        self.bondmap = {i:my_cmap(my_norm(self.edge_sal[i]))[:3] for i in range(len(self.edge_sal))}

        self.highlights = {
            "highlightAtoms": list(self.atommap.keys()),
            "highlightAtomColors": self.atommap,
            "highlightBonds": list(self.bondmap.keys()),
            "highlightBondColors": self.bondmap,
        }
        
    def draw_mol(self, asMol=False, label=None, path='', imgsize=(800, 600)):
        '''
        highlights is a dictionary, which may contains:
        highlightAtoms: list
        highlightBonds: list
        highlightAtomRadii: dict[int]=float, atom index (int), radius (float)
        highlightAtomColors: dict[int]=tuple, index (int), color (tuple, length=3)
        highlightBondColors: dict[int]=tuple,index (int), color (tuple, length=3)
        '''
        smiles = self.smiles
        node_score = self.node_sal_dict
        edge_score = self.edge_sal
        hightlights = self.highlights
        svg_path = path + '_svg'
        os.makedirs(svg_path, exist_ok=True)
        svg_filename = os.path.join(svg_path, self.name + '.svg')
        filename = os.path.join(path, self.name + '.png')
        
        if asMol:
            mol = self.smiles.__copy__()
        else:
            mol = self.mol
        # try:
        mol = rdMolDraw2D.PrepareMolForDrawing(mol)
        # if '.png' in filename:
        #     drawer = rdMolDraw2D.MolDraw2DCairo(*imgsize)
        # else:
        drawer = rdMolDraw2D.MolDraw2DSVG(*imgsize)
        opts = drawer.drawOptions()
        if label == 'map':
            for i in range(mol.GetNumAtoms()):
                opts.atomLabels[i] = mol.GetAtomWithIdx(
                    i).GetSymbol()+str(mol.GetAtomWithIdx(i).GetAtomMapNum())
        if label == 'idx':
            for i in range(mol.GetNumAtoms()):
                opts.atomLabels[i] = mol.GetAtomWithIdx(i).GetSymbol()+str(i)
        if label == 'score':
            assert node_score is not None
            assert edge_score is not None

            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                if idx in self.single_atom_idx:
                    atom.SetProp("atomNote", str(round(node_score[idx], 2)))

            for bond in mol.GetBonds():
                idx = bond.GetIdx()
                if not self.in_same_fg(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()):
                    bond.SetProp("bondNote", str(round(edge_score[idx], 2)))

        if not self.highlights:
            drawer.DrawMolecule(mol)
        else:
            drawer.DrawMolecule(mol, **self.highlights)

        conformer = mol.GetConformer()
        for group in self.group_atom_idx:
            if len(group) == 1:
                continue
            pox = []
            for aid in group:
                pos = conformer.GetAtomPosition(aid)
                pox.append([pos.x, pos.y])
            pox_arr = np.array(pox)
            center = np.mean(pox_arr, axis=0)
            drawer.DrawString(str(round(node_score[group[0]], 2)), Geometry.Point2D(center[0], center[1]))
            
        drawer.FinishDrawing()
        # if '.png' in path:
        #     drawer.WriteDrawingText(path)
            # display(Image(path))
        # else:
        svg = drawer.GetDrawingText()
        # display(SVG(svg.replace('svg:','')))
        if '.svg' in svg_filename:
            with open(svg_filename, 'w') as wf:
                print(svg, file=wf)
            # Convert SVG to PNG
            cairosvg.svg2png(url=svg_filename, write_to=filename)
        return drawer
        # except Exception as e:
        #     print("Check your molecule!!!",e)
        #     return
    
    def draw_mol_atom_level(self, asMol=False, label=None, path='', imgsize=(800, 600)):
        '''
        highlights is a dictionary, which may contains:
        highlightAtoms: list
        highlightBonds: list
        highlightAtomRadii: dict[int]=float, atom index (int), radius (float)
        highlightAtomColors: dict[int]=tuple, index (int), color (tuple, length=3)
        highlightBondColors: dict[int]=tuple,index (int), color (tuple, length=3)
        '''
        smiles = self.smiles
        node_score = self.node_sal_dict
        edge_score = self.edge_sal
        hightlights = self.highlights
        svg_path = path + '_svg'
        os.makedirs(svg_path, exist_ok=True)
        svg_filename = os.path.join(svg_path, self.name + '.svg')
        filename = os.path.join(path, self.name + '.png')
        
        if asMol:
            mol = self.smiles.__copy__()
        else:
            mol = self.mol
        # try:
        mol = rdMolDraw2D.PrepareMolForDrawing(mol)
        # if '.png' in filename:
        #     drawer = rdMolDraw2D.MolDraw2DCairo(*imgsize)
        # else:
        drawer = rdMolDraw2D.MolDraw2DSVG(*imgsize)
        opts = drawer.drawOptions()
        if label == 'map':
            for i in range(mol.GetNumAtoms()):
                opts.atomLabels[i] = mol.GetAtomWithIdx(
                    i).GetSymbol()+str(mol.GetAtomWithIdx(i).GetAtomMapNum())
        if label == 'idx':
            for i in range(mol.GetNumAtoms()):
                opts.atomLabels[i] = mol.GetAtomWithIdx(i).GetSymbol()+str(i)
        if label == 'score':
            assert node_score is not None
            assert edge_score is not None

            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                atom.SetProp("atomNote", str(round(node_score[idx], 2)))
            
            for bond in mol.GetBonds():
                idx = bond.GetIdx()
                bond.SetProp("bondNote", str(round(edge_score[idx], 2)))
            
        if not self.highlights:
            drawer.DrawMolecule(mol)
        else:
            drawer.DrawMolecule(mol, **self.highlights)
        drawer.FinishDrawing()
        # if '.png' in path:
        #     drawer.WriteDrawingText(path)
            # display(Image(path))
        # else:
        svg = drawer.GetDrawingText()
        # display(SVG(svg.replace('svg:','')))
        if '.svg' in svg_filename:
            with open(svg_filename, 'w') as wf:
                print(svg, file=wf)
            # Convert SVG to PNG
            cairosvg.svg2png(url=svg_filename, write_to=filename)
        return drawer
        # except Exception as e:
        #     print("Check your molecule!!!",e)
                             


def draw_saliency_scores(decoding_voc, node_sal_dict, edge_sal_dict, smiles_dict, edge_index_dict, save_path, annotation_type):
    '''
        draw the saliency scores according to the FGs
    '''
    if annotation_type == 3:
        for k, v in tqdm(edge_sal_dict.items()):
            # print('working on ', k)
            drug_sal_instance = drug_sal(k, smiles_dict[k], node_sal_dict[k], v, edge_index_dict[k])
            drug_sal_instance.decomp_fg(decoding_voc)
            drug_sal_instance.compute_sal_score()
            drug_sal_instance.compute_color()
            drug_sal_instance.draw_mol(label='score', path=save_path)
    else:
        for k, v in tqdm(edge_sal_dict.items()):
            # print('working on ', k)
            drug_sal_instance = drug_sal(k, smiles_dict[k], node_sal_dict[k], v, edge_index_dict[k])
            drug_sal_instance.decomp_atom()
            drug_sal_instance.compute_sal_score_atom_level()
            drug_sal_instance.compute_color_atom_level()
            drug_sal_instance.draw_mol_atom_level(label='score', path=save_path)

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
    fig, ax = plt.subplots(figsize=(200, 20))
    sns.set(font_scale=10)


    res = sns.heatmap(values.reshape(1, -1))
    # plt.xticks(np.arange(100), temp[:100])
    res.set_xticklabels(index, 
                        # fontdict={'fontsize':100}, 
                        rotation=90)
    res.set_yticklabels([])

    # plt.show()
    fig.savefig(os.path.join(save_path, name + '.png'), bbox_inches="tight", pad_inches=1)


def draw_gene_saliency(rank_dict, sal_dict, gene_list, save_path, top_n=25):
    i = 0
    for key in tqdm(sal_dict.keys()):
        i += 1
        # print('working on ', key)
        # print('progress: ', i, '/', len(sal_dict))
        rnk = rank_dict[key]
        sal_score = sal_dict[key].reshape(-1)
        ranked_ss = sal_score[rnk]
        ranked_genes = gene_list[rnk]

        draw_one(save_path, key, ranked_ss, ranked_genes, top_n)