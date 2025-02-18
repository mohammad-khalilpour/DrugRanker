import torch
import pdb
import numpy as np
import torch.nn as nn

from models.mpn import MPNN
from models.ae import AE
from torch_geometric.nn import MLP
from utils.common import load_model
from itertools import chain
from pathlib import Path
# import logging

# logger = logging.getLogger("pythonConfig")


class TransformerEncoder(nn.Module):
  def __init__(self, embed_dim, num_heads, forward_expansion, dropout=0.1):
    super(TransformerEncoder, self).__init__()

    self.attention = nn.MultiheadAttention(embed_dim, num_heads)
    self.norm1 = nn.LayerNorm(embed_dim)
    self.norm2 = nn.LayerNorm(embed_dim)

    self.feed_forward = nn.Sequential(
        nn.Linear(embed_dim, forward_expansion*embed_dim),
        nn.ReLU(),
        nn.Linear(forward_expansion*embed_dim, embed_dim)
    )
    
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    attention_out = self.dropout(self.attention(x))
    x = self.norm1(x + attention_out)  
    forward_out = self.dropout(self.feed_forward(x))
    out = self.norm2(x + forward_out)

    return out

class FeatureProjector(nn.Module):
    def __init__(self, channel_list=[], fp_dim=None, emb_dim=None, 
                 in_acts=None, act_last=None):
        super(FeatureProjector, self).__init__()
        self.channel_list = channel_list
        self.in_acts = in_acts
        self.act_last = act_last

        if self.act_last is not None :
            self.last_act_func = nn.Sigmoid()

        if self.in_acts is not None:
            self.hidden_act_fun = nn.ReLU()

        self.out_layer = None
        self.linears = nn.ModuleList()
        if self.channel_list == []:
            self.linears.append(nn.Linear(fp_dim, emb_dim))
        else:
            ch_iterator = zip(self.channel_list[:-1], self.channel_list[1:])
            for in_ch, out_ch in ch_iterator:
                self.linears.append(nn.Linear(in_ch, out_ch))

    def forward(self, x):
        for lins in self.linears[:-1]:
            x = lins(x)
            if self.in_acts is not None :
                x = self.hidden_act_fun(x)

        x = self.linears[-1](x)
        if self.act_last is not None:
            x = self.last_act_func(x)

        if self.out_layer is not None:
            x = self.out_layer(x)
        return x

class Encoder(nn.Module):
	def __init__(self, args):
		super().__init__()
		hidden_size = 4096
		self.encoder = nn.Sequential(
						nn.Linear(args.ae_in_size, hidden_size),
						nn.ReLU(),
						nn.Linear(hidden_size, hidden_size//4),
						nn.ReLU(),
						nn.Linear(hidden_size//4, args.ae_out_size),
						)
        
	def forward(self, x):
		return self.encoder(x)

class Fingerprint(nn.Module):
    def __init__(self, args):
        super(Fingerprint, self).__init__()
        input_dim = 0
        if args.feature_gen == 'rdkit_2d' or args.feature_gen=='rdkit_2d_normalized':
            input_dim = 200
        elif args.feature_gen == 'rdkit_2d_desc':
            input_dim = 210
        elif args.feature_gen == 'morgan' or args.feature_gen == 'morgan_count':
            input_dim = 2048
        elif args.feature_gen == 'rdkit2d_morgan' or args.feature_gen == 'rdkit2d_morganc':
            input_dim = 2048 + 210
        elif args.feature_gen == 'rdkit2d_atompair':
            input_dim = 1024 + 210
        else:
            input_dim = 1024

        if args.update_emb in ["drug-attention"]:
            self.mha = nn.MultiheadAttention(input_dim, 1)
        else:
            self.mha = None
        self.ffn1 = nn.Linear(input_dim, 128)
        self.ffn2 = nn.Linear(128, args.mol_out_size)
        #self.mlp = MLP(channel_list=[input_dim, 256, 128, args.mol_out_size])
        self.relu = nn.ReLU()
        self.device = args.device

    def forward(self, molgraph, features):
        features = torch.from_numpy(np.stack(features)).float().to(self.device)
        if self.mha is not None:
            features, self.drug_weights = self.mha(features, features, features)
        #return self.mlp(features)
        return self.ffn2(self.relu(self.ffn1(features)))


class Scoring(nn.Module):
    def __init__(self, args):
        super(Scoring, self).__init__()
        self.out_size = args.mol_out_size

        self.scoring = args.scoring
        if args.update_emb == 'cell+list-attention2':
            self.scoring = 'mlp2'
            self.ffn = MLP(in_channels=self.out_size ,
                                    hidden_channels=25, num_layers=2, out_channels=1)
        elif args.scoring == 'linear':
            if args.to_use_ae_emb:
                if args.update_emb in ['enc+ppi-attention']:
                    self.ffn = nn.Linear(args.mol_out_size, self.out_size)
                    # self.ffn = nn.Linear(args.ae_out_size*2, self.out_size)
                else:
                    self.ffn = nn.Linear(args.ae_out_size, self.out_size)
            elif args.update_emb in ['ppi-attention']:
                # self.ffn = nn.Linear(args.gene_in_size, self.out_size)
                self.ffn = nn.Linear(args.ae_out_size, self.out_size)
            elif args.update_emb in ['res+ppi-attention']:
                # self.ffn = nn.Linear(args.mol_out_size, self.out_size)
                self.ffn = nn.Linear(args.res_out_size*2, self.out_size)
        elif args.scoring == 'mlp':
            self.ffn = MLP(in_channels=self.out_size+args.ae_out_size ,
                                    hidden_channels=25, num_layers=2, out_channels=1)


    def forward(self, cell_emb, cmp1_emb, cmp2_emb=None, output_type=2):
        """ Type 2: outputs the difference of predicted AUCs (if two input embeddings)
            Type 1: outputs paired scores for paired compounds
            Type 0: only list of scores for list of compounds
        """
        if output_type == 2:
            if self.scoring == 'linear':
                return (self.ffn(cell_emb)*(cmp1_emb - cmp2_emb)).sum(dim=1)
            elif self.scoring == 'mlp':
                return self.ffn(torch.concat((cell_emb, cmp1_emb), dim=1)).squeeze() - \
                self.ffn(torch.concat((cell_emb, cmp2_emb), dim=1)).squeeze() # type: ignore
            #score = (self.ffn(cmp1_emb - cmp2_emb)*cell_emb).sum(dim=1)
        elif output_type == 1:
            if self.scoring == 'linear':
                score1 = (self.ffn(cell_emb)*cmp1_emb).sum(dim=1)
                score2 = (self.ffn(cell_emb)*cmp2_emb).sum(dim=1)
            elif self.scoring == 'mlp':
                score1 = self.ffn(torch.concat((cell_emb, cmp1_emb), dim=1)).squeeze()
                score2 = self.ffn(torch.concat((cell_emb, cmp2_emb), dim=1)).squeeze() # type: ignore
            return score1, score2
        else:
            if self.scoring == 'linear':
                score = (self.ffn(cell_emb)*cmp1_emb).sum(dim=1)
            elif self.scoring == 'mlp':
                score = self.ffn(torch.concat((cell_emb, cmp1_emb), dim=1)).squeeze()
            elif self.scoring == 'mlp2':
                score = self.ffn(cmp1_emb).squeeze()
            #score = (self.scoring(cmp1_emb)*cell_emb).sum(dim=1)
            return score

def sim(x1, x2, sigma=1, kernel='l2'):
    if kernel == 'l2':
        return torch.sum((x1-x2)**2, dim=1)
    if kernel == 'rbf':
        return torch.exp(-sigma*torch.sum((x1-x2)**2, dim=1))


class RankNet(nn.Module):
    def __init__(self, args, mode=None):
        super(RankNet, self).__init__()

        # root_path = Path(args.save_path)
        # file_handler = logging.FileHandler(Path(root_path / f"logs/train_{args.only_fold}.log"), mode="w")
        # logger.addHandler(file_handler)
        self.enc_type = args.gnn

        if args.feature_gen:
            self.enc = Fingerprint(args)
            self.enc_type = args.feature_gen
        elif args.gnn == 'dmpn':
            self.enc = MPNN(args)
            ## over-ride args.mol_out_size
            if 'hier-cat' in args.pooling:
                args.mol_out_size *= args.message_steps
        else:
            raise NotImplementedError

        # for now only MLP update
        self.update_emb = args.update_emb

        self.classify_pairs = args.classify_pairs
        self.classify_cmp = args.classify_cmp
        #self.cluster = args.cluster
        self.agg_emb = args.agg_emb
        self.gene_in_size = args.gene_in_size
        self.to_use_ae_emb = args.to_use_ae_emb

        if self.update_emb == 'concat':
            self.u_mlp = MLP(channel_list=args.mol_out_size*2+args.ae_out_size,
                             hidden_channels=50, num_layers=2, out_channels=args.mol_out_size)
        elif self.update_emb in ['cell-attention', 'sum', 'cell+list-attention']:
            self.u_mlp = MLP(in_channels=args.mol_out_size+args.ae_out_size,
                             hidden_channels=50, num_layers=2, out_channels=args.mol_out_size)
        elif self.update_emb in ['list-attention']:
            self.cell_dim_projector = FeatureProjector(fp_dim=args.ae_out_size, emb_dim=args.mol_out_size)
        elif self.update_emb in ['ppi-attention']:
            self.cell_dim_projector = FeatureProjector(fp_dim=args.gene_in_size, emb_dim=args.ae_out_size,
                                                       in_acts="relu")
        elif self.update_emb in ['enc+ppi-attention']:
            self.u_mlp2 = FeatureProjector(channel_list=[args.gene_in_size, 128, args.ae_out_size], 
                                           in_acts="relu")
            self.cell_dim_projector = MLP(in_channels=args.ae_out_size*2, act="relu",
                                    hidden_channels=50, num_layers=2, out_channels=args.mol_out_size)
        elif self.update_emb in ['res+ppi-attention']:
            self.res_in_size = args.ae_in_size-args.gene_in_size
            self.res_mlp = MLP(channel_list=[self.res_in_size, 4096, 4096//4, args.res_out_size])
            self.u_mlp2 = MLP(channel_list=[args.gene_in_size, 128, args.res_out_size])
            self.cell_dim_projector = MLP(in_channels=args.res_out_size+args.gene_in_size,
                                    hidden_channels=50, num_layers=2, out_channels=args.mol_out_size)

        ## override the mol_out_size again if update rule is concatenation
        if self.update_emb!='None' and args.agg_emb == 'concat':
            args.mol_out_size *= 2

        if self.classify_pairs:
            self.classifierp = MLP(in_channels=args.mol_out_size*2+args.ae_out_size,
                                    hidden_channels=25, num_layers=2, out_channels=1)
        if self.classify_cmp:
            self.classifierc = MLP(in_channels=args.mol_out_size+args.ae_out_size,
                                    hidden_channels=25, num_layers=2, out_channels=1)

        if self.update_emb == 'list-attention':
            self.mha = nn.MultiheadAttention(args.mol_out_size, 5)
        elif 'cell+list-attention' in self.update_emb:
            te_layer = nn.TransformerEncoderLayer(args.mol_out_size, 1, 128)
            self.te = nn.TransformerEncoder(te_layer, 1)
        elif self.update_emb in ['ppi-attention', 'enc+ppi-attention', 'res+ppi-attention']:
            self.cell_mha = nn.MultiheadAttention(args.gene_in_size, 1)
            # te_layer = nn.TransformerEncoderLayer(args.gene_in_size, 1, 128)
            # self.te = nn.TransformerEncoder(te_layer, 1)
        elif self.update_emb in ['attention+enc']:
            self.cell_mha = nn.MultiheadAttention(args.ae_in_size, 1)
        elif self.update_emb in ['drug+ppi-attention']:
            self.cell_drug_mha = nn.MultiheadAttention(args.gene_in_size, 1)
        
        if self.to_use_ae_emb:
            self.ae = AE(args)
            if args.pretrained_ae:
                load_model(self.ae, args.trained_ae_path, args.device)

        self.scoring = Scoring(args)

    def update(self, cell_emb, cmp1_emb, cell_emb2=None, cmp2_emb=None):
        if self.update_emb == 'concat':
            x = torch.concat((cell_emb, cmp1_emb, cmp2_emb), dim=1) + torch.concat((cell_emb, cmp2_emb, cmp1_emb), dim=1)
            c = self.u_mlp(x)
        elif self.update_emb == 'sum':
            x = torch.concat((cell_emb, cmp1_emb+cmp2_emb), dim=1)
            c = self.u_mlp(x)
        elif 'cell' in self.update_emb:
            fused = self.u_mlp(torch.concat((cell_emb, cmp1_emb), dim=1))
            if self.update_emb == 'cell+list-attention':
                cmp1_emb = fused
            elif self.update_emb == 'cell-attention':
                return cmp1_emb
        elif self.update_emb == 'list-attention':
            cell_emb = self.cell_dim_projector(cell_emb)
        elif self.update_emb in ['enc+ppi-attention', 'res+ppi-attention']:
            cell_emb2 = self.u_mlp2(cell_emb2)
            x = torch.concat((cell_emb, cell_emb2), dim=1)
            cell_emb = self.cell_dim_projector(x)
            return cell_emb
        elif self.update_emb == 'ppi-attention':
            cell_emb = self.cell_dim_projector(cell_emb)
            return cell_emb

        if 'cell+list-attention' in self.update_emb:
            output = self.te(cmp1_emb.unsqueeze(dim=1)).squeeze(dim=1)
            return output.squeeze(dim=1)
        elif 'list-attention' in self.update_emb:
            output, weights = self.mha(cell_emb, cmp1_emb, cmp1_emb)
            return output

        gate1 = torch.sigmoid(c*cmp1_emb)
        gate2 = torch.sigmoid(c*cmp2_emb)
        if self.agg_emb == 'sum':
            cmp1_emb = (1+gate1)*cmp1_emb
            cmp2_emb = (1+gate2)*cmp2_emb
        elif self.agg_emb == 'concat':
            cmp1_emb = torch.concat((cmp1_emb, cmp1_emb*gate1), dim=1)
            cmp2_emb = torch.concat((cmp2_emb, cmp2_emb*gate2), dim=1) # type: ignore
        return cmp1_emb, cmp2_emb

    def forward(self, clines, cmp1=None, smiles1=None, feat1=None, 
                clines2=None, cmp2=None, smiles2=None, feat2=None,
                pos=None, neg=None, output_type=2):

        if self.to_use_ae_emb:
            if self.update_emb in ["attention+enc"]:
                cell_emb, self.gene_weights = self.cell_mha(clines, clines, clines)
                cell_emb = self.ae(cell_emb, use_encoder_only=True)
            if self.update_emb in ["enc+ppi-attention"]:
                cell_emb = self.ae(clines.float(), use_encoder_only=True)
                # clines2 = torch.nn.functional.normalize(clines2, dim=0)
                # cell_emb2, self.gene_weights = self.cell_mha(clines2, clines2, clines2)
                cell_emb2 = self.te(clines2)
                self.gene_weights = self.te.state_dict()["layers.0.self_attn.out_proj.weight"] 
            else:
                cell_emb = self.ae(clines.float(), use_encoder_only=True)
        elif self.update_emb in ["ppi-attention", "lasso-attention"]:
            cell_emb, self.gene_weights = self.cell_mha(clines, clines, clines)
            # if torch.isnan(self.gene_weights).any():
            #     logger.info(f"three exists null values in self.gene_weights...")
            # if torch.isnan(cell_emb).any():
            #     logger.info(f"three exists null values in cell_emb...")
            cell_emb2 = clines2
        elif self.update_emb in ["res+ppi-attention"]:
            cell_emb = self.res_mlp(clines)
            cell_emb2, self.gene_weights = self.cell_mha(clines2, clines2, clines2)
        elif self.update_emb in ["drug+ppi-attention"]:
            cell_emb, self.gene_weights = self.cell_mha(clines, clines, clines)
        else:
            cell_emb = clines
            
        plabel = None
        clabel = None
        cmp_sim = None

        if output_type != 0:
            # outputs difference of scores or paired
            if pos and neg:
                cmp_emb = self.enc(cmp1, feat1)
                cmp1_emb = cmp_emb[pos]
                cmp2_emb = cmp_emb[neg]
            else:
                cmp1_emb = self.enc(cmp1, feat1)
                cmp2_emb = self.enc(cmp2, feat2)

            if self.update_emb != 'None':
                cmp1_emb, cmp2_emb = self.update(cell_emb, cmp1_emb, cmp2_emb)

            if self.classify_pairs:
                plabel = self.classifierp(torch.concat((cell_emb, cmp1_emb, cmp2_emb),dim=1)).squeeze() # type: ignore

            if self.classify_cmp:
                clabel1 = self.classifierc(torch.concat((cell_emb, cmp1_emb),dim=1)).squeeze()
                clabel2 = self.classifierc(torch.concat((cell_emb, cmp2_emb),dim=1)).squeeze() # type: ignore
                clabel = torch.concat((clabel1, clabel2))

            #if self.cluster:
            #    cmp_sim = sim(cmp1_emb, cmp2_emb)

            return self.scoring(cell_emb, cmp1_emb, cmp2_emb, output_type), plabel, clabel, cmp_sim

        else:
            cmp_emb = self.enc(cmp1, feat1)

            if self.update_emb != 'None':
                if self.update_emb in ['ppi-attention', 'enc+ppi-attention', 'res+ppi-attention']:
                    cell_emb = self.update(cell_emb, cmp_emb, cell_emb2=cell_emb2)
                elif self.update_emb in ['attention+enc', 'drug-attention']:
                    pass
                else:
                    cmp_emb = self.update(cell_emb, cmp_emb)

            return self.scoring(cell_emb, cmp_emb, output_type=output_type)

