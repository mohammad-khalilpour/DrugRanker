from pathlib import Path
from utils.args import parse_args

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from dataloader.loader import to_batchgraph
from utils.common import set_seed
from models.ranknet import RankNet
from dataloader.loader import CellLine, MoleculePoint, MoleculeDatasetTrain, MoleculeDatasetTest
from dataloader.utils import *
from utils.common import *

SEED = 123


def get_dataloader(args, data, splits, thresholds):
    sfold = 0
    kfold = args.kfold
    print(f"kfold: {kfold}")
    
    if args.only_fold != -1:
        sfold = args.only_fold
        kfold = args.only_fold+1
    
    data_loaders = []
    for fold in range(sfold, kfold):
        print(f"fold: {fold}")
        # set seeds for reproducibility
        set_seed()
        threshold = thresholds[fold] if thresholds else None
        if splits:
            split = splits[fold]
            train_index, val_index, test_index = split['train'], split['val'], split['test']
        else:
            # no splits provided, raise error
            raise ValueError("No splits provided for cross-validation.")
            #train_index, test_index = list(range(len(dataset)*4//5)), list(range(len(dataset)*4//5, len(dataset)))

        clobj = CellLine(args.genexp_path)

        train_auc, val_auc, test_auc = get_fold_data_LRO(data, train_index, val_index, test_index, args.smiles_path) \
                                if (args.setup == 'LRO') else \
                                get_fold_data_LCO(data, train_index, val_index, test_index, args.smiles_path)

        features = precompute_features(args) # returns feature_dict keyed by smiles
        train_pts, val_pts, test_pts = [], [], []
        # train_pts and test_pts are lists of MoleculePoint objects where each object stores drug and cell line pair attributes
        for d in train_auc:
            train_pts.append(MoleculePoint(*d, features=features[d[0]], feature_gen=args.feature_gen, in_test=False))
        for d in val_auc:
            val_pts.append(MoleculePoint(*d, features=features[d[0]], feature_gen=args.feature_gen, in_test=True))
        for d in test_auc:
            test_pts.append(MoleculePoint(*d, features=features[d[0]], feature_gen=args.feature_gen, in_test=True))

        # not required for ListOne and ListAll
        train_ps, test_ps = get_pair_setting(args)

        train_threshold = threshold['train'] if args.setup == 'LCO' else threshold
        train_dataset = MoleculeDatasetTrain(train_pts, delta=args.delta, threshold=train_threshold,
                                            num_pairs=args.num_pairs, pair_setting=train_ps,
                                            sample_list=args.sample_list, mixture=args.mixture)
        train_threshold = train_dataset.threshold
        val_threshold = train_threshold if (args.setup == 'LRO') else threshold['val'] # for val eval
        test_threshold = train_threshold if (args.setup == 'LRO') else threshold['test'] # for test eval

        val_dataset = MoleculeDatasetTest(val_pts, threshold=val_threshold)
        test_dataset = MoleculeDatasetTest(test_pts, threshold=test_threshold)
        traineval_dataset = MoleculeDatasetTest(train_pts, threshold=train_threshold) if args.do_train_eval else None
        # create combined dataset for 1st experimental setup
        combeval_dataset = MoleculeDatasetTest(train_pts+test_pts, threshold=None) if args.do_comb_eval else None
        total_dataset = MoleculeDatasetTest(list(set(train_pts+val_pts+test_pts)), threshold=None) if args.do_test else None

        # create dataloaders
        train_dataloader = DataLoader(train_dataset, shuffle=True, #pin_memory=True,
                            batch_size = args.batch_size, collate_fn = train_dataset.collate_fn)
        val_dataloader = DataLoader(val_dataset, shuffle=False, #pin_memory=True,
                        batch_size = args.batch_size, collate_fn = test_dataset.collate_fn)
        test_dataloader = DataLoader(test_dataset, shuffle=False, #pin_memory=True,
                        batch_size = args.batch_size, collate_fn = test_dataset.collate_fn)

        if args.do_train_eval:
            traineval_dataloader = DataLoader(traineval_dataset, shuffle=False, #pin_memory=True, # type: ignore
                        batch_size = args.batch_size, collate_fn = traineval_dataset.collate_fn)
        if args.do_comb_eval:
            combeval_dataloader = DataLoader(combeval_dataset, shuffle=False, #pin_memory=True, # type: ignore
                        batch_size = args.batch_size, collate_fn = combeval_dataset.collate_fn)
        if args.do_test:
            total_dataloader = DataLoader(total_dataset, shuffle=False, #pin_memory=True, # type: ignore
                        batch_size = args.batch_size, collate_fn = total_dataset.collate_fn)
            
        logs_dir = os.path.join(args.save_path, 'logs')
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        if (not os.path.exists(args.save_path + f'fold_{fold+1}')):
            os.makedirs(args.save_path + f'fold_{fold+1}')

        # # output intermediate metrics per cell line
        # f_val = open(os.path.join(logs_dir, f'val_{fold+1}.txt'), 'w')
        # f_test = open(os.path.join(logs_dir, f'test_{fold+1}.txt'), 'w')
        # f_json = open(os.path.join(logs_dir, f'scores_{fold+1}.json'), 'w')
        # if args.do_train_eval:
        #     f_train = open(os.path.join(logs_dir, f'train_{fold+1}.txt'), 'w')
        # if args.do_comb_eval:
        #     f_comb = open(os.path.join(logs_dir, f'train+test_{fold+1}.txt'), 'w')
        # if args.do_test:
        #     pass
        data_loaders.append((fold+1, total_dataloader))
    return clobj, data_loaders

def get_model(args, model_path):
    model = RankNet(args)
    checkpoint = torch.load(os.path.join(model_path), map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)
    model = model.to(args.device)
    return model

def get_feat_vector(input, model, target_layer="cell_mha"):
    # input_tensor = preprocess(input_image)
    # input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        my_output = None
        
        def my_hook(module_, input_, output_):
            nonlocal my_output
            my_output = output_

        if target_layer=="cell_mha":
            a_hook = model.cell_mha.register_forward_hook(my_hook)        
        model(input["clines"], cmp1=input["cmp1"], smiles1=input["smiles1"],
            feat1=input["feat1"], clines2=input["clines2"], output_type=0)
        a_hook.remove()
        return my_output

def predict(args, clobj, model, test_dataloader, verbose_mha=False):
    true_auc = []
    preds = []
    aw_outputs = {}
    ccl_ids = []
    cpd_ids = []
    labels = []
    in_test = []	
    model.eval()
    with tqdm(test_dataloader, unit="batch") as bar:
        for batch in bar:
            bar.set_description(f"Evaluating")
            mols, features, clids = [], [], []

            for d in batch:
                true_auc.append(d.auc)
                mols.append(d.smiles)
                features.append(d.features)
                ccl_ids.append(d.clid)
                clids.append(d.clid)
                labels.append(d.label)
                in_test.append(d.in_test)

            # print(f"clids: {clids}")
            # print(f"features: {features[0].shape}")
            gene_expression = np.array(clobj.get_expression(clids))
            cl_emb = torch.from_numpy(gene_expression).to(args.device).to(args.device)
            cl_emb2 = None
            if args.update_emb in ["ppi-attention"]:
                selected_gindices = np.load(args.selected_genexp_path)
                cl_emb = torch.from_numpy(gene_expression[:, selected_gindices]).to(args.device)
            elif args.update_emb in ["enc+ppi-attention"]:
                selected_gindices = np.load(args.selected_genexp_path)
                cl_emb2 = torch.from_numpy(gene_expression[:, selected_gindices]).to(args.device).to(args.device)
            elif args.update_emb in ["res+ppi-attention"]:
                selected_gindices = np.load(args.selected_genexp_path)
                res_indices = np.delete(np.arange(gene_expression.shape[1]), selected_gindices)
                cl_emb = torch.from_numpy(gene_expression[:, res_indices]).to(args.device)
                cl_emb2 = torch.from_numpy(gene_expression[:, selected_gindices]).to(args.device)

            cpd_ids += [d.cpdid for d in batch]

            molgraph = to_batchgraph(mols) if args.gnn else None

            input = {"clines": cl_emb,
                     "cmp1": molgraph,
                     "smiles1": mols,
                     "feat1": features,
                     "clines2": cl_emb2,
                     }

            if clids[0] not in aw_outputs:
                aw_outputs[clids[0]] = list(get_feat_vector(input, model)[0].to("cpu").numpy().flatten())
            if verbose_mha:
                print(f"{clids} weights: {np.shape(aw_outputs[clids])}")
                print(f"{clids} weights: {aw_outputs[clids]}")
            pred = model(cl_emb, cmp1=molgraph, smiles1=mols, feat1=features,
                         clines2=cl_emb2, output_type=0).data.cpu().flatten().tolist()
            preds.extend(pred)
            # bar.set_postfix(
            #     loss = float(total_loss/iters)
            # )

    pred_dict = None
	# metrics, m_clid, pred_dict = compute_metrics(true_auc, preds, ccl_ids, labels, in_test, cpd_ids, Kpos)
    return preds, true_auc, pred_dict, aw_outputs #, metrics, m_clid

def main(args):
    splits, thresholds = None, None
    if (args.setup == 'LRO'):
        data, splits, thresholds = get_cv_data_LRO(args.data_path, args.splits_path)
    elif (args.setup == 'LCO'):
        data, splits, thresholds = get_cv_data_LCO(args.data_path, args.splits_path)
    else:
        raise ValueError('Invalid setup')

    args.device = torch.device(args.desired_device if torch.cuda.is_available() and args.cuda else "cpu")

    ## data loading
    clobj, data_loaders = get_dataloader(args, data, splits, thresholds)

    ## selected ccle df
    ccle_df = pd.read_csv(args.genexp_path, index_col=0)
    gene_indices = np.load(args.selected_genexp_path)
    selected_genes = np.array(list(ccle_df.columns))[gene_indices]
    print(selected_genes)
    selected_genes = [c.split(" ")[0] for c in selected_genes]

    for item in data_loaders:
        fold, dloader = item
        print(dloader)
        ## model loading
        fold_path = Path(args.save_path) / f'fold_{fold}' 
        model_path = list(fold_path.rglob("./epoch_*.pt"))[0]
        print(f"model_path: {model_path}")
        gene_aw_path = list(fold_path.rglob("./gene_aw_epoch*.npy"))[0]
        gene_aw = np.load(gene_aw_path)
        model = get_model(args, model_path)
        print(f"model: {model}")

        ## get prediction and attention weights
        outputs = predict(args, clobj, model, dloader, verbose_mha=False)
        aw_dict = outputs[3]
        aw_df = pd.DataFrame(aw_dict).T
        aw_df.columns = selected_genes
        print(aw_df)
        aw_df.to_csv(str(fold_path / "cell_aw.csv"))
        mean_caws = list(aw_df.mean().sort_values(ascending = False).index)
        # np.save(str(fold_path / "mean_caws.npy"), mean_caws)
        with open(str(fold_path / "mean_caws.txt"), 'w') as f:
            for g in mean_caws:
                f.write("%s\n" % g)

if __name__ == "__main__":
    # args = parser.parse_args()
    args = parse_args()
    main(args)