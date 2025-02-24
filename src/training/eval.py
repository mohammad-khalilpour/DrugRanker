import numpy as np
import torch
from utils.metrics import *
from utils.common import *
from features.featurization import mol2graph
from dataloader.loader import to_batchgraph

from collections import defaultdict
import time

def evaluate(clobj, model, test_dataloader, args, Kpos):
	model.eval()

	true_auc = []
	preds = []
	ccl_ids  = []
	cpd_ids = []
	labels   = []
	in_test = []

	for batch in test_dataloader:
		mols, features, clids = [], [], []

		for d in batch:
			true_auc.append(d.auc)
			mols.append(d.smiles)
			features.append(d.features)
			ccl_ids.append(d.clid)
			clids.append(d.clid)
			labels.append(d.label)
			in_test.append(d.in_test)

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
			
		cpd_ids += [d.cpdid for d in batch]

		molgraph = to_batchgraph(mols) if args.gnn else None

		pred = model(cl_emb, cmp1=molgraph, smiles1=mols, feat1=features, 
			   		clines2=cl_emb2, output_type=0).data.cpu().flatten().tolist()
		preds.extend(pred)

	pred_dict = None
	metrics, m_clid, pred_dict = compute_metrics(true_auc, preds, ccl_ids, labels, in_test, cpd_ids, Kpos)
	return preds, true_auc, metrics, m_clid, pred_dict