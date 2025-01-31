# XGDP

## Environment

```
conda env create --file=environment.yml
```

---

## Data Preparation

### Download the raw data

The data used in this study is available in our [Google drive](https://drive.google.com/drive/folders/1n-30SyGfHbdV08H_cGKjFCqJ9fSnL4DV?usp=sharing).

If you want to use the latest dataset, download the drug response data in IC50 format called PANCANCER_IC from [GDSC](https://www.cancerrxgene.org/downloads/drug_data). And download the gene expression data called CCLE_expression from [CCLE](https://depmap.org/portal/download/all/) under mRNA expression. 

### Preprocess the data

- Create a folder in your project directory called `root_folder`.
``` bash
mkdir root_folder
```

- Place the PANCANCER_IC data under folder `data/GDSC` and place the CCLE_expression data under folder `data/CCLE`. Choose a `<branch_num>` as you like and run the following command to preprocess the data. 
The data will be saved under `root_folder/<branch_num>`.
``` python
python load_data.py <branch_num>
```

## Train the model

``` python
python train.py \
        --model <model_num>
        --branch <branch_num>
        --do_cv
        --do_attn
```
- Available models: 0:GCN, 1:GAT, 2:GAT_Edge, 3:GATv2, 4:SAGE, 5:GIN, 6:GINE, 7:WIRGAT, 8:ARGAT, 9:RGCN, 10:FiLM

## Explain the model

Instead of training the models from scratch, you can use the pretrained models under `models/`. Place them under `root_folder/<branch_num>` where you stored the processed data in [Preprocess the data](###Preprocess the data)

### Attribute the chemical structures with GNNExplainer

``` python
python gnnexplainer.py \
        --model <model_num>
        --branch <branch_num>
        --do_attn
        --explain_type <type>
python draw_gnnexplainer.py \
        --model <model_num>
        --branch <branch_num>
        --explain_type <type>
        --annotation <type>
```
- Available explaining types: 0:model, 1:phenomenon
- Available annotation types: 0:numbers, 1:heatmap, 2:both, 3:functional group-level heatmap
    - Numbers: Only the saliency scores will be visualized 
    - Heatmap: The atom-level heatmap to show saliency levels
    - Both: Both numbers and heatmaps will be displayed
    - Functional group-level heatmap **(Recommended)**: The saliency scores are accumulated for each functional groups rather than atoms. Both numbers and heatmaps will be displayed with this mode. 

### Attribute the gene expression values with Integrated Gradients

``` python
python integrated_gradients.py \
        --model <model_num>
        --branch <branch_num>
        --do_attn
        --iqr_baseline
```
#### Pathway Analysis

- Download the gene sets from [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp) and place them under `data/`.
- Refer to `pathway_analysis.ipynb` for the pathway analysis experiments based on the gene saliency scores.