import numpy as np
import pandas as pd
import torch
import pandas as pd
import torch.nn as nn
import pickle
from tool.my_dataset import get_dataset_dataloader
from tool.dacon_eval import evaluate_scores, calculate_score
from tool.my_model import HawonNet
np.set_printoptions(precision=6, suppress=True)

class args:
    def __init__(self):
        self.n_gnn = 3
        self.dim_gnn = 128
        self.ngpu = 1
        self.test_data_dir = '/home/tech/Hawon/Dacon/data/toygnn_data/train_test_dump'
        self.num_workers = 2
        self.num_epochs = 3001
        self.dropout_rate = 0.1
        self.interaction_net = True
        self.dev_vdw_radius = 0.2
        self.lr = 1e-3
        self.distance_bins=15
        self.residue_node=False
        self.gnn_layer_type = 'AttentiveFP' # GCN or AttentiveFP
        self.int_projection_dim = 32
hawon_args = args()

with open('/home/tech/Hawon/Dacon/data/toygnn_data/id_to_y.pkl', 'rb') as fp:
    id_to_y = pickle.load(fp)
with open('/home/tech/Hawon/Dacon/data/toygnn_data/public_test_keys.pkl', 'rb') as fp:
    public_test_keys = pickle.load(fp)
with open('/home/tech/Hawon/Dacon/data/toygnn_data/private_test_keys.pkl', 'rb') as fp:
    private_test_keys = pickle.load(fp)
    
    
def pIC50_to_IC50(pIC50): # IC50 -> nM 단위
    return 10**(-pIC50+9)

def public_test_eval(public_test_keys, data_dir, id_to_y, ckpt_path, mode):
    test_dataset, test_dataloader = get_dataset_dataloader(keys=public_test_keys,
                                                            data_dir=data_dir,
                                                            id_to_y=id_to_y,
                                                            batch_size=1,
                                                            num_workers=1,
                                                            residue_node=hawon_args.residue_node,
                                                            train=False,
                                                            inference=True)
    model = HawonNet(hawon_args)
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
                                                            residue_node=hawon_args.residue_node,
                                                            train=False,
                                                            inference=True)
    model = HawonNet(hawon_args)
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
    
public_test_eval(public_test_keys = public_test_keys,
                 data_dir = hawon_args.test_data_dir,
                 id_to_y = id_to_y,
                 ckpt_path = '/home/tech/Hawon/Dacon/code/MAIN/trial5_attentiveFP_ToyGNN/ToyGNN/ckpt/best_model.pth',
                 mode = 'score') # mode : score / pred


private_test_eval(private_test_keys = private_test_keys,
                  data_dir = hawon_args.test_data_dir,
                  id_to_y = id_to_y,
                  ckpt_path = '/home/tech/Hawon/Dacon/code/MAIN/trial5_attentiveFP_ToyGNN/ToyGNN/ckpt/best_model.pth',
                  load_path = '/home/tech/Hawon/Dacon/data/raw/sample_submission.csv',
                  save_path = f'./{input("원하는 파일명을 입력하세요")}.csv') 