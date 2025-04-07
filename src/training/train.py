import numpy as np
import torch
import torch.nn as nn
from torchviz import make_dot

from dataloader.loader import to_batchgraph
from features.features_generators import *
from features.featurization import mol2graph
from utils.nn_utils import compute_pnorm, compute_gnorm
from utils.common import pair2set

from tqdm import tqdm
from pathlib import Path

import time

def train_step_listnet(clobj, model, loader, criterion, optimizer, epoch, args):
    """
    train listwise models
    `clobj`: cell line object
    `model`: model object
    `loader`: data loader
    `criterion`: loss function
    `optimizer`: optimizer  
    """
    model.train()
    total_loss = 0
    iters, gnorm = 0, 0

    model.zero_grad()
    with tqdm(loader, unit="batch") as bar:
        for batch in bar:
            bar.set_description(f"Epoch {epoch}")
            batch_loss = 0
            clids, mols, features, labels, aucs = [], [], [], [], []
            for d in batch:
                aucs.append(d.auc)
                mols.append(d.smiles)
                features.append(d.features)
                clids.append(d.clid)
                labels.append(d.label)

            gene_expression = np.array(clobj.get_expression(clids))
            normalized_gene_expression = np.array(clobj.get_normalized_expression(clids))
            cl_emb = torch.from_numpy(gene_expression).to(args.device)
            cl_emb2 = None
            if args.update_emb in ["ppi-attention", "drug+ppi-attention"]:
                selected_gindices = np.load(args.selected_genexp_path)
                cl_emb = torch.from_numpy(normalized_gene_expression[:, selected_gindices]).to(args.device)
            elif args.update_emb in ["enc+ppi-attention"]:
                selected_gindices = np.load(args.selected_genexp_path)
                cl_emb2 = torch.from_numpy(normalized_gene_expression[:, selected_gindices]).to(args.device)
            elif args.update_emb in ["res+ppi-attention"]:
                selected_gindices = np.load(args.selected_genexp_path)
                res_indices = np.delete(np.arange(gene_expression.shape[1]), selected_gindices)
                cl_emb = torch.from_numpy(gene_expression[:, res_indices]).to(args.device)
                cl_emb2 = torch.from_numpy(gene_expression[:, selected_gindices]).to(args.device)
                
            # batch graph needed only for gnn models
            molgraph = to_batchgraph(mols) if args.gnn else None

            # print(np.shape(features))
            pred = model(cl_emb, cmp1=molgraph, smiles1=mols, feat1=features, 
                         clines2=cl_emb2, output_type=0)

            plot_path = "/media/external_16TB_1/kian_khalilpour/DrugRanker/assets/model_graph/cell_drug_attention"
            if not Path(plot_path).exists():
                make_dot(pred, params=dict(list(model.named_parameters()))).render(plot_path, format="png")

            if args.model == 'listone' :
                batch_loss = criterion(pred, torch.tensor(aucs, device=pred.device))
            elif args.model == 'listall':
                batch_loss = criterion(pred.reshape(1,-1), torch.tensor(labels, device=pred.device).reshape(1,-1))
            elif args.model == 'lambdarank' or args.model == 'neuralndcg' or args.model == 'lambdaloss' or args.model == 'approxndcg':
                batch_loss = criterion(pred.reshape(1,-1), torch.tensor(aucs, device=pred.device).reshape(1,-1))
            else:
                raise ValueError('Invalid listwise model name')
            total_loss += batch_loss.item()

            batch_loss.backward()
            if (iters+1) == len(loader) or (iters+1)%args.gradient_steps==0:
                optimizer.step()
                gnorm = compute_gnorm(model)
                optimizer.zero_grad()
            iters += 1
            bar.set_postfix(
                loss = float(total_loss/iters)
            )

    return total_loss/iters, gnorm


def train_step(clobj, model, loader, criterion, optimizer, epoch, args):
    """
    train pairwise models
    `clobj`: cell line object
    `model`: model object
    `loader`: data loader
    `criterion`: loss function
    `optimizer`: optimizer
    """
    model.train()
    total_loss = 0
    iters, gnorm = 0, 0
    labels = None
    bce = nn.BCEWithLogitsLoss()

    with tqdm(loader, unit="batch") as bar:
        for batch in bar:
            bar.set_description(f"Epoch {epoch}")
            ccl_ids = []
            mols1, mols2, features1, features2, labels1, labels2, aucs1, aucs2 = [], [], [], [], [], [], [], []
            for d1, d2 in batch:
                ccl_ids.append(d1.clid)
                mols1.append(d1.smiles)
                features1.append(d1.features)
                labels1.append(d1.label)
                aucs1.append(d1.auc)

                mols2.append(d2.smiles)
                features2.append(d2.features)
                labels2.append(d2.label)
                aucs2.append(d2.auc)

            cl_emb = torch.from_numpy(np.asarray(clobj.get_expression(ccl_ids))).to(args.device)
            # sign = 1 if 1st comp is more sensitive than the 2nd comp; else -1
            sign = torch.from_numpy(np.sign(np.array(aucs2) - np.array(aucs1))).to(args.device)
            # y = 1 if both the comp in a pair are of same label, else 0
            y = torch.from_numpy(np.array(np.array(labels1) == np.array(labels2), dtype=int)).to(args.device)

            if args.model == 'pairpushc':
                # to reduce call to batch and self.gnn, convert pairs to sets of graphs and features
                pos, neg, list_mols, list_features = pair2set(batch)
                molgraph = to_batchgraph(list_mols) if args.gnn else None 

                pred_diff, plabel, clabel, cmp_sim = model(cl_emb, cmp1=molgraph,
                                                        smiles1=list_mols, feat1=list_features,
                                                            pos=pos, neg=neg)
            else:
                raise ValueError('Invalid model name')

            batch_loss = 0

            #actual_diff = torch.from_numpy(np.array(sens1) - np.array(sens2)).to(args.device)
            if args.surrogate == 'logistic':
                # sign should be 1 if (+,-) pair and -1 if (-,+) pair
                batch_loss = torch.mean(criterion(-sign*pred_diff), dim=0)
            elif args.surrogate == 'tcbb':
                batch_loss = criterion(pred_diff, labels1, labels2, sign)
            else:
                raise ValueError('Invalid surrogate loss name')

            #elif args.surrogate == 'margin':
            #    batch_loss += criterion(pred_diff, y, sign)

            # drug pair classification loss: not used in paper
            if args.classify_pairs:
                bce_loss = bce(plabel, y.float())
                batch_loss += bce_loss

            # drug instance sensitivity classification loss
            if args.classify_cmp:
                clabel = clabel.flatten()
                labels = torch.Tensor(np.array(labels1 + labels2)).to(clabel.device)
                batch_loss += bce(clabel, labels.float())

            ## regularization
            if args.regularization:
                batch_loss = batch_loss + args.regularization*compute_pnorm(model)**2

            batch_loss.backward()
            if (iters+1) == len(loader) or (iters+1)%args.gradient_steps==0:
                optimizer.step()
                gnorm = compute_gnorm(model)
                model.zero_grad()

            total_loss += batch_loss.item()
            iters += 1
            
            bar.set_postfix(
                loss = float(total_loss/iters)
            )

    return total_loss/iters, gnorm
