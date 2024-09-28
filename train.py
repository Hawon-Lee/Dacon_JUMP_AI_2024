import os
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import argparse
from tool.my_dataset import get_dataset_dataloader
from tool.dacon_eval import evaluate_scores, calculate_score
from tool.my_model import HawonNet
from tool.etc import pearson_correlation, pIC50_to_IC50
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

def main():
    parser = argparse.ArgumentParser(description='Training script for HawonNet')
    parser.add_argument('--total_data_dir', type=str, help='Training data directory.')
    parser.add_argument('--tr_keys', type=str, help='File path of train keys(pkl)')
    parser.add_argument('--vl_keys', type=str, help='File path of validation keys(pkl)')
    parser.add_argument('--id_to_y', type=str, help='File path of id_to_y dictionary(pkl)')
    parser.add_argument('--gnn_type', type=str, default='GCN', choices=['GCN', 'AttentiveFP'], help='The type of gnn architecture.')
    parser.add_argument('--gnn_n_layer', type=str, default=3)
    parser.add_argument('--gnn_hidden_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--use_residue_node', action='store_true', help='Use whole graph whose protein node is c_alpha')
    parser.add_argument('--distance_bins', type=int, default=15, help='Num of distance bins to make edge features between to node(atom)')  
    parser.add_argument('--int_projection_dim', type=int, default=32, help='Dimension to project edge feature')    
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--save_model', type=str, default=True)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--train_log_dir', type=str)
    args = parser.parse_args()
    
    with open(args.tr_keys, 'rb') as fp:
        tr_keys = pickle.load(fp)
    with open(args.vl_keys, 'rb') as fp:
        vl_keys = pickle.load(fp)
    with open(args.id_to_y, 'rb') as fp:
        id_to_y = pickle.load(fp)
    
    
    # Training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HawonNet(args)
    if args.ckpt:
        ckpt = torch.load('/home/tech/Hawon/Dacon/code/MAIN/trial5_attentiveFP_ToyGNN/ToyGNN/ckpt/save/pretrained_AttentiveFP_resnodeFalse.pth')
        model.load_state_dict(ckpt)
    model.to(device)
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 스케쥴러 / 얼리스탑 정의
    scheduler = ReduceLROnPlateau(optim, mode='max', factor=0.1, patience=5)
    patience = 15
    min_delta = 0.001

    # best model save
    best_val_dacon = -1
    best_val_mse = np.Inf
    best_model_path = './ckpt/best_model.pth'

    train_dataset, train_dataloader = get_dataset_dataloader(keys = tr_keys, 
                                                            data_dir = args.total_data_dir, 
                                                            id_to_y = id_to_y, 
                                                            batch_size = args.batch_size, 
                                                            num_workers = args.num_workers, 
                                                            residue_node = args.use_residue_node,
                                                            train = True,
                                                            inference=False) # train -> shuffle 여부

    val_dataset, val_dataloader = get_dataset_dataloader(keys = vl_keys, 
                                                        data_dir = args.total_data_dir, 
                                                        id_to_y = id_to_y, 
                                                        batch_size = args.batch_size, 
                                                        num_workers = args.num_workers, 
                                                        residue_node = args.use_residue_node,
                                                        train = False,
                                                        inference=False)

    print(f'Data load done: train {len(train_dataset)}, val {len(val_dataset)}')

    tr_mse, vl_mse, tr_pc, vl_pc, best_result = [], [], [], [], {}
    tr_dacon_score, vl_dacon_score = [], []

    # 학습 루프

    print(f'''OPTIONS
    - epoch: {args.num_epochs}
    - batch_size: {args.batch_size}
    - num_workers: {args.num_workers}
    - lr: {args.lr}
    - gnn_type : {args.gnn_type}
    - gnn_n_layer: {args.gnn_n_layer}
    - gnn_h_dim : {args.gnn_hidden_dim}
    - residue_node : {args.use_residue_node}
    =================================
    ''')

    for epoch in range(args.num_epochs):
        # train
        model.train()
        train_loss = 0.0
        train_labels = torch.tensor([])
        train_predicts = torch.tensor([])
        for batch_idx, sample in enumerate(tqdm(train_dataloader)):
            # put sample on device
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    value = value.to(device)
                    sample[key] = value
            optim.zero_grad()
            train_pred = model(sample).squeeze()
            loss = criterion(train_pred, sample['affinity'])
            loss.backward()
            optim.step()
            train_loss += loss.item() * sample['affinity'].size(0)
            train_labels = torch.concat([train_labels, sample['affinity'].cpu()])
            train_predicts = torch.concat([train_predicts, train_pred.cpu()])
            torch.cuda.empty_cache()

        # validation
        model.eval()
        val_loss = 0.0
        val_labels = torch.tensor([])
        val_predicts = torch.tensor([])
        with torch.no_grad():
            for batch_idx, sample in enumerate(tqdm(val_dataloader)):
                # put sample on device
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        value = value.to(device)
                        sample[key] = value
                val_pred = model(sample).squeeze()
                loss = criterion(val_pred, sample['affinity'])
                val_loss += loss.item() * sample['affinity'].size(0)
                val_labels = torch.cat([val_labels, sample['affinity'].cpu()])
                val_predicts = torch.cat([val_predicts, val_pred.cpu()])

        # MSE/PC 계산
        train_labels = train_labels.detach().numpy()
        train_predicts = train_predicts.detach().numpy()
        val_labels = val_labels.detach().numpy()
        val_predicts = val_predicts.detach().numpy()
        
        train_mse = train_loss / len(train_dataset)
        val_mse = val_loss / len(val_dataset)
        train_pearson = pearson_correlation(train_labels, train_predicts).item()
        val_pearson = pearson_correlation(val_labels, val_predicts).item()
        train_dacon = calculate_score(pIC50_to_IC50(train_labels), pIC50_to_IC50(train_predicts)).item()
        val_dacon = calculate_score(pIC50_to_IC50(val_labels), pIC50_to_IC50(val_predicts)).item()
        # train_dacon = evaluate_scores(pIC50_to_IC50(train_labels), pIC50_to_IC50(train_predicts))[0].item()
        # val_dacon = evaluate_scores(pIC50_to_IC50(val_labels), pIC50_to_IC50(val_predicts))[0].item()
        
        # scheduler step
        scheduler.step(val_dacon)
        
        # 로깅
        print(f'epoch:{epoch+1} | train_mse(pic50):{train_mse:.2f} | val_mse(pic50):{val_mse:.2f} | train_dacon(ic50): {train_dacon:.2f} | val_dacon(ic50): {val_dacon:.2f} | train_pearson:{train_pearson:.2f} | val_pearson:{val_pearson:.2f}')
        tr_mse.append(train_mse)
        vl_mse.append(val_mse)
        tr_pc.append(train_pearson)
        vl_pc.append(val_pearson)
        tr_dacon_score.append(train_dacon)
        vl_dacon_score.append(val_dacon)
        
        # 모델 저장 & early stop
        if args.save_model:
            if val_dacon >= best_val_dacon + min_delta:
                best_val_dacon = val_dacon
                torch.save(model.state_dict(), best_model_path)
                print(f'New best model saved based on val_dacon -> epoch: {epoch+1}, val_mse: {val_mse:.2f}, val_dacon: {best_val_dacon:.2f}')
                best_result['best_val_mse'] = best_val_mse
                best_result['best_val_dacon'] = best_val_dacon
                best_result['epoch'] = epoch+1
                counter = 0
            elif val_mse <= best_val_mse - min_delta:
                best_val_mse = val_mse
                torch.save(model.state_dict(), best_model_path)
                print(f'New best model saved based on val_mse -> epoch: {epoch+1}, val_mse: {best_val_mse:.2f}, val_dacon: {val_dacon:.2f}')
                best_result['best_val_mse'] = best_val_mse
                best_result['best_val_dacon'] = best_val_dacon
                best_result['epoch'] = epoch+1
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopped at epoch {epoch + 1} !')
                    break
        print(f'-----------------------------------------------------------------')

    with open('train_logs/tr_mse.pkl', 'wb') as fp:
        pickle.dump(tr_mse, fp)
    with open('train_logs/vl_mse.pkl', 'wb') as fp:
        pickle.dump(vl_mse, fp)
    with open('train_logs/tr_pc.pkl', 'wb') as fp:
        pickle.dump(tr_pc, fp)
    with open('train_logs/vl_pc.pkl', 'wb') as fp:
        pickle.dump(vl_pc, fp)
    with open('train_logs/best_result.pkl', 'wb') as fp:
        pickle.dump(best_result, fp)

if __name__ == '__main__':
    main()