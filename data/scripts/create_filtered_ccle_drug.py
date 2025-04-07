import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path

parser = ArgumentParser('gene drug selection mapping.')
parser.add_argument('--data_dir', help='Path to the data directory.',
                    default='data/ctrp/LRO/')
parser.add_argument('--gene_drug_file', help='Path to the gene/drug mapping files.',
                    default='data/ctrp/selected_genes_ctrp.csv')
parser.add_argument('--ccle_file', help='Path to the gene expression data.',
                    default='data/CCLE/CCLE_expression.csv')
parser.add_argument('--gene_file', help='Path to the gene indices.',
                    default='data/ctrp/LRO/genes.txt')
parser.add_argument('--drug_file', help='Path to the drug indices.',
                    default='data/ctrp/LRO/drugs.txt')

def to_exist_gene(target, df, return_index=True):
    if return_index:
        indices = np.array([target == g.split(" ")[0] for g in list(df.columns)])
        # indices = np.array([target in g for g in list(df.columns)])
    else:
        indices = _
    return any(indices), indices

def main(args):

    # ## get ccle file cell-line info
    ccle_df = pd.read_csv(args.ccle_file, index_col=0)
    # c_exists, indices = to_exist_drug("CYP11A1", gene_exp_df)
    # print(c_exists)
    # print(f"the exact name of the gene is {np.array(list(gene_exp_df.columns))[indices]}")
    print(ccle_df.info)
    print(len(ccle_df.columns))
    # print(gene_exp_df["CYP11A1"])

    ## get selected gene/drug info
    gd_df = pd.read_csv(args.gene_drug_file, index_col=0)
    print(gd_df)
    selected_genes = sorted(set(gd_df["genes"].values))
    print(f"selected_genes: {len(selected_genes)}")
    selected_drugs = sorted(set(gd_df["drugs"].values))
    print(f"selected_drugs: {len(selected_drugs)}")

    ## get drug indices 
    dindices = pd.read_csv(args.drug_file, sep=" ", header=None)
    print(f"dindices: {len(dindices)}")

    missing_genes = []
    selected_genes_cnames = []
    indices_list = []
    ## preparing selected genes
    for g in tqdm(selected_genes):
        g_exists, indices = to_exist_gene(g, ccle_df)
        # print(sum(indices.astype("int")))
        # print(f"the exact name of the gene is {np.array(list(ccle_df.columns))[indices][0]}")
        if sum(indices.astype("int")) == 0:
            missing_genes.append(g)
        else:
            assert sum(indices.astype("int")) == 1, f"there is a problem with the gene: {g}"
            selected_genes_cnames.append(np.array(list(ccle_df.columns))[indices][0])
            indices_list.append(np.where(indices==True)[0][0])
    print(f"missing_genes: {len(missing_genes)}")
    print(f"missing_genes: {missing_genes}")
    with open('data/ctrp/missing_genes_ctrp.txt', 'w') as f:
        for g in missing_genes:
            f.write("%s\n" % g)

    np.save('data/ctrp/selected_genes_indices_ctrp.npy', np.array(indices_list))
            
    print(ccle_df[selected_genes_cnames])
    ccle_df_cleaned = ccle_df[selected_genes_cnames]
    ccle_new_dir = Path(args.ccle_file).parent
    ccle_new_fname = str(Path(args.ccle_file).stem)+"_ctrp_w20.csv"
    ccle_new_fpath = Path(ccle_new_dir) / ccle_new_fname
    ccle_df_cleaned.to_csv(str(ccle_new_fpath))
    print(pd.read_csv(ccle_new_fpath, index_col=0))


if __name__ == "__main__":
    args=parser.parse_args()
    main(args)