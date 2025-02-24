import sys
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader


from utils.args import parse_args
from utils.common import *
from utils.nn_utils import compute_pnorm, compute_gnorm, param_count, EarlyStopping
from models.ranknet import RankNet
from models.loss import (PairPushLoss, ListOneLoss, ListAllLoss, LambdaLoss, 
                            lambdaRank_scheme, NeuralNDCG, ApproxNDCGLoss)
from dataloader.loader import CellLine, MoleculePoint, MoleculeDatasetTrain, MoleculeDatasetTest
from dataloader.utils import *

from training.train import train_step_listnet, train_step
from training.eval import evaluate


SEED = 123
# logger = logging.getLogger('train.log')
# logger.setLevel(logging.DEBUG)

# # Print to the terminal
# log_level = logging.DEBUG
# logging.root.setLevel(log_level)
# formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
# stream = logging.StreamHandler()
# stream.setLevel(log_level)
# stream.setFormatter(formatter)
# log = logging.getLogger("pythonConfig")
# if not log.hasHandlers():
#     log.setLevel(log_level)
#     log.addHandler(stream)

# # file handler:
# root_path
# file_handler = logging.FileHandler(Path(root_path / "train_.log"), mode="w")
# file_handler.setLevel(log_level)
# file_handler.setFormatter(formatter)
# log.addHandler(file_handler)

# Print to the terminal
log_level = logging.DEBUG
logging.root.setLevel(log_level)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
stream = logging.StreamHandler()
stream.setLevel(log_level)
stream.setFormatter(formatter)
logger = logging.getLogger("pythonConfig")
if not logger.hasHandlers():
    logger.setLevel(log_level)
    logger.addHandler(stream)

def log_metrics(metric, mode, epoch, fold=None):
    # print("mode", "epoch", "fold", sep=',', end=',')
    # for k, v in metric.items():
    #     print('%s' %k, end=',')
    # print()
    print(mode, epoch, fold, sep=',', end=',')
    for k, v in metric.items():
        if fold:
            logger.info('%s (Epoch %d) : %s for fold %d = %.4f' %(mode, epoch, k, fold, v))
        else:
            logger.info('%s (Epoch %d) : Avg %s = %.4f' %(mode, epoch, k, v))
        print('%.4f' %v, end=',')
    print()


def run(model, dataset, train_index, val_index, test_index, threshold,
        criterion, optimizer, fold, args):
    """
    Run training and evaluation for a single fold.
    `model`: a RankNet object
    `dataset`: a MoleculeDatasetTrain object
    `train_index`: a list of indices for the training set
    `test_index`: a list of indices for the test set
    `threshold`: a dict of thresholds for the training and test sets
    `criterion`: a loss function
    `optimizer`: an optimizer
    `fold`: the fold number
    `args`: a Namespace object
    """
    val_metrics = {}
    train_metrics = {}
    test_metrics = {}
    comb_metrics = {}
    METRICS = ['CI', 'lCI', 'sCI', 'ktau', 'sp']
    Kpos = [1,3,5,10,20,40,60]
    for k in Kpos:
        METRICS += [f'AP@{k}', f'AH@{k}', f'NDCG@{k}']

    clobj = CellLine(args.genexp_path)

    train_auc, val_auc, test_auc = get_fold_data_LRO(dataset, train_index, val_index, test_index, args.smiles_path) \
                            if (args.setup == 'LRO') else \
                            get_fold_data_LCO(dataset, train_index, val_index, test_index, args.smiles_path)


    # print(f"train_index: {train_index[0]}")
    # print(f"train_auc: {train_auc[0]}")
    features = precompute_features(args) # returns feature_dict keyed by smiles

    normalized_features = {}
    feat_values = np.array(list(features.values()))
    min_values = feat_values.min(axis=0)
    max_values = feat_values.max(axis=0)
    normalized_feat_values = (feat_values - min_values) / (max_values - min_values)
    normalized_feat_values = np.nan_to_num(normalized_feat_values)
    for i, durg_feats in enumerate(features.items()):
        k, v = durg_feats
        normalized_features[k] = list(normalized_feat_values[i])

    train_pts, val_pts, test_pts = [], [], []
    # train_pts and test_pts are lists of MoleculePoint objects where each object stores drug and cell line pair attributes
    for d in train_auc:
        train_pts.append(MoleculePoint(*d, features=features[d[0]], feature_gen=args.feature_gen, in_test=False))
    for d in val_auc:
        val_pts.append(MoleculePoint(*d, features=features[d[0]], feature_gen=args.feature_gen, in_test=True))
    for d in test_auc:
        test_pts.append(MoleculePoint(*d, features=features[d[0]], feature_gen=args.feature_gen, in_test=True))
    # print(f"train_pts: {len(train_pts)}")
    # print(f"train_pts: {train_pts[0]}")

    # not required for ListOne and ListAll
    train_ps, test_ps = get_pair_setting(args)

    logger.info(f'#Train AUC: {len(train_auc)}, #Val AUC: {len(val_auc)}, #Test AUC: {len(test_auc)}')
    logger.info(f'Pair settings: {train_ps}: train, {test_ps}: test/val')
    logger.info(f'Num Pairs: {args.num_pairs}')

    train_threshold = threshold['train'] if args.setup == 'LCO' else threshold
    train_dataset = MoleculeDatasetTrain(train_pts, delta=args.delta, threshold=train_threshold,
                                         num_pairs=args.num_pairs, pair_setting=train_ps,
                                         sample_list=args.sample_list, mixture=args.mixture)
    train_threshold = train_dataset.threshold
    val_threshold = train_threshold if (args.setup == 'LRO') else threshold['val'] # for val eval
    test_threshold = train_threshold if (args.setup == 'LRO') else threshold['test'] # for test eval

    val_dataset       = MoleculeDatasetTest(val_pts, threshold=val_threshold)
    test_dataset      = MoleculeDatasetTest(test_pts, threshold=test_threshold)
    traineval_dataset = MoleculeDatasetTest(train_pts, threshold=train_threshold) if args.do_train_eval else None
    # create combined dataset for 1st experimental setup
    combeval_dataset = MoleculeDatasetTest(train_pts+test_pts, threshold=None) if args.do_comb_eval else None

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

    logs_dir = os.path.join(args.save_path, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    if (not os.path.exists(args.save_path + f'fold_{fold}')):
        os.makedirs(args.save_path + f'fold_{fold}')

    # output intermediate metrics per cell line
    f_val = open(os.path.join(logs_dir, f'val_{fold}.txt'), 'w')
    f_test = open(os.path.join(logs_dir, f'test_{fold}.txt'), 'w')
    f_json = open(os.path.join(logs_dir, f'scores_{fold}.json'), 'w')
    if args.do_train_eval:
        f_train = open(os.path.join(logs_dir, f'train_{fold}.txt'), 'w')
    if args.do_comb_eval:
        f_comb = open(os.path.join(logs_dir, f'train+test_{fold}.txt'), 'w')

    json_out = []
    early_stop = EarlyStopping(patience=args.log_steps)
    
    best_result = -1
    for epoch in range(1, args.max_iter+1):
        ## TODO: need to be reconsidered
        if args.model in ['listone', 'listall', 'lambdarank', 'neuralndcg', 'lambdaloss', 'approxndcg']:
            loss, gnorm = train_step_listnet(clobj, model, train_dataloader, criterion, optimizer, epoch, args)
        else:
            loss, gnorm = train_step(clobj, model, train_dataloader, criterion, optimizer, epoch, args)

        logger.info("Loss at epoch = %d : %.4f" %(epoch, loss))
        logger.info("Pnorm at epoch = %d : %.4f" %(epoch, compute_pnorm(model)))
        logger.info("GNorm at epoch = %d : %.4f" %(epoch, gnorm))

        early_stop.step(compute_pnorm(model))
        # logging and evaluation at every `log_steps`
        if (epoch) and (epoch % args.log_steps == 0):
            # save models 
            if (epoch==args.max_iter) or (args.checkpointing and (epoch % args.log_steps == 0)):
                torch.save(model.state_dict(), args.save_path +f'fold_{fold}/epoch_{epoch}.pt')
                if os.path.exists(args.save_path +f'fold_{fold}/epoch_{epoch - args.log_steps}.pt'):
                    os.remove(args.save_path +f'fold_{fold}/epoch_{epoch - args.log_steps}.pt') # one file at a time
                if args.to_save_attention_weights:
                    if args.update_emb in ["drug-attention"]:
                        np.save(args.save_path +f'fold_{fold}/drug_aw_epoch_{epoch}.npy', model.enc.drug_weights.detach().cpu().numpy())
                        if os.path.exists(args.save_path +f'fold_{fold}/drug_aw_epoch_{epoch - args.log_steps}.npy'):
                            os.remove(args.save_path +f'fold_{fold}/drug_aw_epoch_{epoch - args.log_steps}.npy') # one file at a time
                                        
                    elif args.update_emb in ["drug+ppi-attention"]:
                        np.save(args.save_path +f'fold_{fold}/drug_aw_epoch_{epoch}.npy', model.enc.drug_weights.detach().cpu().numpy())
                        if os.path.exists(args.save_path +f'fold_{fold}/drug_aw_epoch_{epoch - args.log_steps}.npy'):
                            os.remove(args.save_path +f'fold_{fold}/drug_aw_epoch_{epoch - args.log_steps}.npy') # one file at a time
                        
                        np.save(args.save_path +f'fold_{fold}/gene_aw_epoch_{epoch}.npy', model.gene_weights.detach().cpu().numpy())
                        if os.path.exists(args.save_path +f'fold_{fold}/gene_aw_epoch_{epoch - args.log_steps}.npy'):
                            os.remove(args.save_path +f'fold_{fold}/gene_aw_epoch_{epoch - args.log_steps}.npy') # one file at a time
             
                    else:
                        np.save(args.save_path +f'fold_{fold}/gene_aw_epoch_{epoch}.npy', model.gene_weights.detach().cpu().numpy())
                        if os.path.exists(args.save_path +f'fold_{fold}/gene_aw_epoch_{epoch - args.log_steps}.npy'):
                            os.remove(args.save_path +f'fold_{fold}/gene_aw_epoch_{epoch - args.log_steps}.npy') # one file at a time

            pred_scores, true_auc, metric, m_clid, pred_dict = evaluate(clobj, model, val_dataloader, args, Kpos)
            log_metrics(metric, 'VAL', epoch, fold)
            for k,v in m_clid.items():
                st = [f'{v[_]:.4f}' for _ in METRICS]
                print(str(epoch) + ',' + k + ',' + ','.join(st), file=f_val)
            val_metrics[epoch] = metric
            json_out.append({'VAL:'+str(epoch): pred_dict})

            pred_scores, true_auc, metric, m_clid, pred_dict = evaluate(clobj, model, test_dataloader, args, Kpos)
            log_metrics(metric, 'TEST', epoch, fold)
            for k,v in m_clid.items():
                st = [f'{v[_]:.4f}' for _ in METRICS]
                print(str(epoch) + ',' + k + ',' + ','.join(st), file=f_test)

            if args.to_save_best:
                if metric['NDCG@10'] > best_result:
                    if os.path.exists(args.save_path +f'fold_{fold}/best.pt'):
                        os.remove(args.save_path +f'fold_{fold}/best.pt') # one file at a time
                    torch.save(model.state_dict(), args.save_path +f'fold_{fold}/epoch_best.pt')
                    if args.to_save_attention_weights:
                        if args.update_emb in ["drug-attention"]:
                            if os.path.exists(args.save_path +f'fold_{fold}/drug_aw_best.npy'):
                                os.remove(args.save_path +f'fold_{fold}/drug_aw_best.npy') # one file at a time
                            np.save(args.save_path +f'fold_{fold}/drug_aw_best.npy', model.enc.drug_weights.detach().cpu().numpy())
                        elif args.update_emb in ["drug+ppi-attention"]:
                            if os.path.exists(args.save_path +f'fold_{fold}/drug_aw_best.npy'):
                                os.remove(args.save_path +f'fold_{fold}/drug_aw_best.npy') # one file at a time
                            np.save(args.save_path +f'fold_{fold}/drug_aw_best.npy', model.enc.drug_weights.detach().cpu().numpy())

                            if os.path.exists(args.save_path +f'fold_{fold}/gene_aw_best.npy'):
                                os.remove(args.save_path +f'fold_{fold}/gene_aw_best.npy') # one file at a time
                            np.save(args.save_path +f'fold_{fold}/gene_aw_best.npy', model.gene_weights.detach().cpu().numpy())
                        else:
                            if os.path.exists(args.save_path +f'fold_{fold}/gene_aw_best.npy'):
                                os.remove(args.save_path +f'fold_{fold}/gene_aw_best.npy') # one file at a time
                            np.save(args.save_path +f'fold_{fold}/gene_aw_best.npy', model.gene_weights.detach().cpu().numpy())
                    best_result = metric['NDCG@10']
                    logger.info('The best model saved at (Epoch %d) with %s = %.4f' %(epoch, 'NDCG@10', best_result))
            test_metrics[epoch] = metric
            json_out.append({'TEST:'+str(epoch): pred_dict})

            if args.do_train_eval:
                pred_scores, true_auc, metric, m_clid, _ = evaluate(clobj, model, traineval_dataloader, args, Kpos)
                log_metrics(metric, 'TRAIN', epoch, fold)
                for k,v in m_clid.items():
                    st = [f'{v[_]:.4f}' for _ in METRICS]
                    print(str(epoch) + ',' + k + ',' + ','.join(st), file=f_train)
                train_metrics[epoch] = metric

            if args.do_comb_eval:
                pred_scores, true_auc, metric, m_clid, _ = evaluate(clobj, model, combeval_dataloader, args, Kpos)
                log_metrics(metric, 'TRAIN+TEST', epoch, fold)
                for k,v in m_clid.items():
                    st = [f'{v[_]:.4f}' for _ in METRICS]
                    print(str(epoch) + ',' + k + ',' + ','.join(st), file=f_comb)
                comb_metrics[epoch] = metric

        if early_stop.stop:
            logger.info("Early Stopping... weights converged...")
            break

    json.dump(json_out, f_json)
    f_test.close()
    f_json.close()
    if args.do_train_eval:
        f_train.close()
    if args.do_comb_eval:
        f_comb.close()

    return val_metrics, test_metrics, train_metrics, comb_metrics


def cross_validate(args, dataset, splits=None, thresholds=None):
    """
    Cross validate the model
    `dataset` is a dictionary of `cell_ID, drug_ID: auc`  
    `splits` is a list of dictionary with keys (train, test)
    """
    
    args.device = torch.device(args.desired_device if torch.cuda.is_available() and args.cuda else "cpu")

    argparse_dict = vars(args)
    if args.device == torch.device('cpu'):
        argparse_dict['device'] = 'cpu'
    else:
        argparse_dict['device'] = args.desired_device

    if args.only_fold <= 0:
        with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
            json.dump(argparse_dict, fjson)

    # print command line
    logger.debug(f'python {" ".join(sys.argv)}')
    sfold = 0
    kfold = args.kfold

    train_metrics_folds, val_metrics_folds, test_metrics_folds = [], [], []
    comb_metrics_folds = []

    if args.only_fold != -1:
        sfold = args.only_fold
        kfold = args.only_fold+1

    for fold in range(sfold, kfold):
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

        # initialize the model
        ## TODO: need to be reconsidered
        if args.model in ['pairpushc', 'listone', 'listall', 'lambdaloss',
                            'lambdarank', 'neuralndcg', 'approxndcg']:
            model = RankNet(args)
        else:
            # many other models which were implemented but not used in the paper
            raise NotImplementedError

        model = model.to(args.device)
        logger.debug(model)
        #print(model)

        optimizer = opt.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

        if args.model == 'listone':
            criterion = ListOneLoss(args.M)
        elif args.model == 'listall':
            criterion = ListAllLoss(args.M)
        elif args.model == 'pairpushc':
            criterion = PairPushLoss(args.alpha, args.beta)
        elif args.model == 'lambdaloss':
            criterion = LambdaLoss()
        elif args.model == 'lambdarank':
            criterion = LambdaLoss(weighing_scheme=lambdaRank_scheme)
        elif args.model == 'neuralndcg':
            criterion = NeuralNDCG()
        elif args.model == 'approxndcg':
            criterion = ApproxNDCGLoss()
        else:
            ## many other models are implemented but not required for the workshop paper
            raise NotImplementedError 
    
        logger.info('Total Params Count: %d' %(param_count(model)))
        logger.info('Cross validation: Fold %d/%d' %(fold+1, 5))

        if args.do_train:
            logger.info('Start training...')
            m1, m2, m3, m4 = run(model, dataset, train_index, val_index, test_index, threshold,
                             criterion, optimizer, fold+1, args)
            val_metrics_folds.append(m1)
            test_metrics_folds.append(m2)
            train_metrics_folds.append(m3)
            comb_metrics_folds.append(m4)

    # report average performance across folds for every `log_steps`
    for ep in range(args.log_steps, args.max_iter+1, args.log_steps):
        avg_metrics = []
        for fold in range(kfold-sfold):
            for e, metric in val_metrics_folds[fold].items():
                if e == ep:
                    avg_metrics.append(metric)
        results_dict = calc_avg_perf(avg_metrics)
        log_metrics(results_dict, 'VAL', ep)
        results_dict["mode"] = 'VAL'
        results_dict["epoch"] = ep
        print(results_dict)

    for ep in range(args.log_steps, args.max_iter+1, args.log_steps):
        avg_metrics = []
        for fold in range(kfold-sfold):
            for e, metric in test_metrics_folds[fold].items():
                if e == ep:
                    avg_metrics.append(metric)
        results_dict = calc_avg_perf(avg_metrics)
        log_metrics(results_dict, 'TEST', ep)
        results_dict["mode"] = 'TEST'
        results_dict["epoch"] = ep
        print(results_dict)

    if args.do_train_eval:
        for ep in range(args.log_steps, args.max_iter+1, args.log_steps):
            avg_metrics = []
            for fold in range(kfold-sfold):
                for e, metric in train_metrics_folds[fold].items():
                    if e == ep:
                        avg_metrics.append(metric)
            log_metrics(calc_avg_perf(avg_metrics), 'TRAIN', ep)

    if args.do_comb_eval:
        for ep in range(args.log_steps, args.max_iter+1, args.log_steps):
            avg_metrics = []
            for fold in range(kfold-sfold):
                for e, metric in comb_metrics_folds[fold].items():
                    if e == ep:
                        avg_metrics.append(metric)
            log_metrics(calc_avg_perf(avg_metrics), 'TRAIN+TEST', ep)


def main(args):
    splits, thresholds = None, None
    if (args.setup == 'LRO'):
        data, splits, thresholds = get_cv_data_LRO(args.data_path, args.splits_path)
    elif (args.setup == 'LCO'):
        data, splits, thresholds = get_cv_data_LCO(args.data_path, args.splits_path)
    else:
        raise ValueError('Invalid setup')

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # file handler:
    root_path = Path(args.save_path)
    file_handler = logging.FileHandler(Path(root_path / f"logs/train_{args.only_fold}.log"), mode="w")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # print(f"data: {data['train']}")
    # print(f"data: {data.keys()}")
    # print(f"splits: {splits[0]['train']}")
    cross_validate(args, data, splits, thresholds)


if __name__ == '__main__':
    main(parse_args())
