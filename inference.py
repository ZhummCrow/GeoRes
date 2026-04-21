import numpy as np
import os, random, pickle
from tqdm import tqdm
import torch.nn as nn
from torch_geometric.loader import DataLoader
from model import *
import pandas as pd
import argparse
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric

class ProteinGraphTestDataset(data.Dataset):
    def __init__(self, dataset, index, args):
        super(ProteinGraphTestDataset, self).__init__()       
        index = list(set(index))
        self.emb_path = args.feature_path+"saprot/"
        self.radius = args.radius
        self.dataset = dataset.loc[index]
        self.dataset.reset_index(drop=True, inplace=True)
        self.protein_name = 'kpc'
        self.X = torch.load(args.feature_path+"pdbs/"+self.protein_name+".tensor")
        self.DSSP = torch.load(args.feature_path+"pdbs/"+self.protein_name+"_DSSP.tensor")
        X_ca = self.X[:, 1]
        self.edge_index = torch_geometric.nn.radius_graph(X_ca, r=self.radius, loop=True, max_num_neighbors = 1000, num_workers = 4)
        
    def __len__(self): return len(self.dataset)

    def get_csv(self): return self.dataset
    
    def __getitem__(self, idx): return self._featurize_graph(idx)
    
    def _featurize_graph(self, idx):
        row = self.dataset.loc[idx]
        mut_seq_id = row['prot.geno']
        delta_df_index = row['index'] 
        with torch.no_grad():
            mut_esm = torch.load(self.emb_path + mut_seq_id + "_raw.tensor")
            mut_pre_computed_node_feat = torch.cat([mut_esm, self.DSSP], dim=-1)  
            wt_esm = torch.load(self.emb_path+ "WT_raw.tensor")
            wt_pre_computed_node_feat = torch.cat([wt_esm, self.DSSP], dim=-1)  
        mut_graph_data = torch_geometric.data.Data(name=delta_df_index,  node_feat=mut_pre_computed_node_feat, 
                                                   X=self.X,edge_index=self.edge_index)
        wt_graph_data = torch_geometric.data.Data(name=delta_df_index,  node_feat=wt_pre_computed_node_feat, )
        return wt_graph_data, mut_graph_data

def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_and_predict(model_class, config, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert args.output_path[-4:] == ".csv", "output_path must end with xxx.csv"
    
    node_input_dim = 1280 + 9 + 184
    edge_input_dim = 450
    hidden_dim = 128
    layer = 2
    augment_eps = 0
    dropout = 0
    num_workers = 8
    task_list = config['task']
    folds = config['folds']
    
    dms_df = pd.read_csv(args.dataset_path)
    test_dataset = ProteinGraphTestDataset(dms_df,list(range(len(dms_df))), args)
    test_dataloader = DataLoader(test_dataset, batch_size = config['batch_size'], shuffle=False, drop_last=False, num_workers=num_workers, prefetch_factor=2)
    
    models = []
    for fold in range(folds):
        state_dict = torch.load(args.model_path + 'fold%s.ckpt'%fold, device)
        model = model_class(node_input_dim, edge_input_dim, hidden_dim, layer, dropout, augment_eps,task_list).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    test_pred_dict = {} # 导出测试结果
    for data in tqdm(test_dataloader):
        wt_data,mut_data = data[0].to(device),data[1].to(device)
        with torch.no_grad():
            outputs = [model.inference(wt_data,mut_data) for model in models] # [[b,num task,3]*5]
            outputs_class = [res[1] for res in outputs]
            outputs_class = torch.stack(outputs_class,0).mean(0).detach().cpu().numpy() # 5个模型预测结果求平均
        indexs = wt_data.name # name是数据集中的index
        for i, index in enumerate(indexs):
            test_pred_dict[index] = outputs_class[i]
    for drugi,drug in enumerate(task_list):
        dms_df[f'{drug}_class']=[test_pred_dict[index][drugi].tolist() for index in dms_df['index'].tolist()]
    dms_df.to_csv(args.output_path,index=False)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default='../inference_data/kpc_df.csv')
parser.add_argument("--feature_path", type=str, default='../inference_data/')
parser.add_argument("--output_path", type=str, default='./output/inference/xxx.csv')
parser.add_argument("--model_path", type=str, default="./weights/seed1/")
parser.add_argument("--bs", type=int, default=24)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--hid", type=int, default=128)
args = parser.parse_args()
Seed_everything(42)
model_class = GPSite
nn_config = {
    'task':["IMP_rg","MEM_rg","CZA_rg","AZA_rg"],
    'radius':12,
    'batch_size': args.bs,
    'folds': 5,
}

if __name__ == '__main__':
    train_and_predict(model_class, nn_config, args)
