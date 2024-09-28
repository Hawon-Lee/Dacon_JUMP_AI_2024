import numpy as np
import pandas as pd
import torch
import pandas as pd
import torch.nn as nn
import pickle
import argparse
from tool.my_dataset import get_dataset_dataloader
from tool.dacon_eval import evaluate_scores, calculate_score
from tool.my_model import HawonNet
np.set_printoptions(precision=6, suppress=True)

# training option과 동일하게 맞춰주세요...
def main():
    parser = argparse.ArgumentParser(description='Training script for HawonNet')
    parser.add_argument('--test_data_dir', type=str, help='Test data directory.')
    parser.add_argument('--test_keys', type=str, help='File path of test keys(pkl)')
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
    parser.add_argument('--ckpt', type=str, help='Model checkpoint to evaluate')
    parser.add_argument('--train_log_dir', type=str)
    args = parser.parse_args()


    with open(args.id_to_y, 'rb') as fp:
        id_to_y = pickle.load(fp)
    with open(args.test_keys, 'rb') as fp:
        public_test_keys = pickle.load(fp)
        
    def pIC50_to_IC50(pIC50): # IC50 -> nM 단위
        return 10**(-pIC50+9)

    def public_test_eval(public_test_keys, data_dir, id_to_y, ckpt_path, mode):
        test_dataset, test_dataloader = get_dataset_dataloader(keys=public_test_keys,
                                                                data_dir=data_dir,
                                                                id_to_y=id_to_y,
                                                                batch_size=1,
                                                                num_workers=1,
                                                                residue_node=args.use_residue_node,
                                                                train=False,
                                                                inference=True)
        model = HawonNet(args)
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint)
        
        submit_id = []
        submit_pIC = []
        answer_pIC = []

        model.eval()
        with torch.no_grad():
            for sample in test_dataloader:
                submit_id.append(sample['key'][0])
                submit_pIC.append(model(sample).item())
                answer_pIC.append(id_to_y[sample['key'][0]])
                
        submit_IC = pIC50_to_IC50(np.array(submit_pIC))
        answer_IC = pIC50_to_IC50(np.array(answer_pIC))
        if mode not in ['score', 'pred']:
            raise Exception('score 또는 pred를 인자로 넣거라')
        if mode == 'score':
            return calculate_score(answer_IC, submit_IC)
        else:
            return submit_IC
        
    def private_test_eval(private_test_keys, data_dir, id_to_y, ckpt_path, load_path, save_path):
        test_dataset, test_dataloader = get_dataset_dataloader(keys=private_test_keys,
                                                                data_dir=data_dir,
                                                                id_to_y=id_to_y,
                                                                batch_size=1,
                                                                num_workers=1,
                                                                residue_node=args.residue_node,
                                                                train=False,
                                                                inference=True)
        model = HawonNet(args)
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint)

        submit_id = []
        submit_pIC = []

        model.eval()
        with torch.no_grad():
            for sample in test_dataloader:
                submit_id.append(sample['key'][0][1:])
                submit_pIC.append(model(sample).item())
                
        submit_IC = pIC50_to_IC50(np.array(submit_pIC))
        
        submit = pd.read_csv(load_path)
        submit['IC50_nM'] = submit_IC
        submit.to_csv(save_path, index=False)
        print(f'result saved at {save_path}')
        
    print(public_test_eval(public_test_keys = public_test_keys,
                    data_dir = args.test_data_dir,
                    id_to_y = id_to_y,
                    ckpt_path = args.ckpt,
                    mode = 'score')
          )# mode : score / pred

if __name__ == '__main__':
    main()
    


# private_test_eval(private_test_keys = private_test_keys,
#                 data_dir = args.test_data_dir,
#                 id_to_y = id_to_y,
#                 ckpt_path = '/home/tech/Hawon/Dacon/code/MAIN/trial5_attentiveFP_ToyGNN/ToyGNN/ckpt/best_model.pth',
#                 load_path = '/home/tech/Hawon/Dacon/data/raw/sample_submission.csv',
#                 save_path = f'./{input("원하는 파일명을 입력하세요")}.csv') 