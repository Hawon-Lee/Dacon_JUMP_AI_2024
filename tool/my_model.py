import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import GRUCell, Linear, Parameter
import torch.nn.functional as F

from torch_geometric.nn import models, global_mean_pool
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool
from torch_geometric.nn.models.attentive_fp import GATEConv
from torch_geometric.data import Data, Batch

from torch_scatter import scatter_mean, scatter_sum, scatter_max


class custom_AttentiveFP(torch.nn.Module):
    r"""The Attentive FP model for molecular representation learning from the
    `"Pushing the Boundaries of Molecular Representation for Drug Discovery
    with the Graph Attention Mechanism"
    <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ paper, based on
    graph attention mechanisms.
    
    ** Since it is custom, this layer will only embed node level and do not aggregate them into graph level. **
    
    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global
            readout.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)

    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.lin1 = Linear(in_channels, hidden_channels)

        self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim,
                                  dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)

        self.atom_convs = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        self.mol_conv = GATConv(hidden_channels, hidden_channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_conv.explain = False  # Cannot explain global pooling.
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin1.reset_parameters()
        self.gate_conv.reset_parameters()
        self.gru.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
                batch: Tensor) -> Tensor:
        """"""  # noqa: D419
        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x))

        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = conv(x, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu()

        return self.lin2(x)
        # # Molecule Embedding:
        # row = torch.arange(batch.size(0), device=batch.device)
        # edge_index = torch.stack([row, batch], dim=0)

        # out = global_add_pool(x, batch).relu_()
        # for t in range(self.num_timesteps):
        #     h = F.elu_(self.mol_conv((x, out), edge_index))
        #     h = F.dropout(h, p=self.dropout, training=self.training)
        #     out = self.mol_gru(h, out).relu_()

        # # Predictor:
        # out = F.dropout(out, p=self.dropout, training=self.training)
        return self.lin2(out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'out_channels={self.out_channels}, '
                f'edge_dim={self.edge_dim}, '
                f'num_layers={self.num_layers}, '
                f'num_timesteps={self.num_timesteps}'
                f')')
    
class HawonNet(nn.Module):
    def __init__(self, hawon_args, dropout_rate=0.1):
        super().__init__()
        self.args = hawon_args
        self.relu = nn.ReLU()
        
        if self.args.use_residue_node:
            n_feat = 30
        else: n_feat = 54
        
        self.l_node_embedding = nn.Linear(54, hawon_args.gnn_hidden_dim, bias=False)
        self.t_node_embedding = nn.Linear(n_feat, hawon_args.gnn_hidden_dim, bias=False)
        
        # if self.args.gnn_layer_type == 'GCN':
        #     gcn_params = {'in_channels': hawon_args.dim_gnn,
        #         'hidden_channels': hawon_args.dim_gnn,
        #         'out_channels': hawon_args.dim_gnn,
        #         'num_layers': hawon_args.n_gnn}
        #     self.l_gconv = models.GCN(**gcn_params)
        #     self.t_gconv = models.GCN(**gcn_params)
        # elif self.args.gnn_layer_type == 'AttentiveFP':
        #     afp_params = {'in_channels': hawon_args.dim_gnn,
        #         'hidden_channels': hawon_args.dim_gnn,
        #         'out_channels': hawon_args.dim_gnn,
        #         'edge_dim': hawon_args.distance_bins,
        #         'num_layers': hawon_args.n_gnn,
        #         'num_timesteps': 2,
        #         'dropout': 0.1}
        #     self.l_attentiveFP = custom_AttentiveFP(**afp_params)
        #     self.t_attentiveFP = custom_AttentiveFP(**afp_params)
        # else: raise Exception('arguments name_error : gnn_layer_type must be "GCN" or "AttentiveFP"')
        
        gcn_params = {'in_channels': hawon_args.gnn_hidden_dim,
            'hidden_channels': hawon_args.gnn_hidden_dim,
            'out_channels': hawon_args.gnn_hidden_dim,
            'num_layers': hawon_args.gnn_n_layer}
        afp_params = {'in_channels': hawon_args.gnn_hidden_dim,
            'hidden_channels': hawon_args.gnn_hidden_dim,
            'out_channels': hawon_args.gnn_hidden_dim,
            'edge_dim': hawon_args.distance_bins,
            'num_layers': hawon_args.gnn_n_layer,
            'num_timesteps': 2,
            'dropout': 0.1}
        self.l_gconv = models.GCN(**gcn_params)
        self.t_gconv = models.GCN(**gcn_params)
        self.l_attentiveFP = custom_AttentiveFP(**afp_params)
        self.t_attentiveFP = custom_AttentiveFP(**afp_params)
        
        self.int_projection_layer = nn.Linear(hawon_args.distance_bins, hawon_args.int_projection_dim, bias=False)

        self.int_mlp = nn.Sequential(
                        nn.Linear(hawon_args.gnn_hidden_dim * 2 + hawon_args.int_projection_dim, hawon_args.gnn_hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(hawon_args.gnn_hidden_dim*2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )        
        
        
        self.w_h = nn.Linear(hawon_args.gnn_hidden_dim, 1, bias=False) # attention score for each interaction edge's hidden state
    
    
    def get_edge_features(self, adj, pos): # 거리를 15개로 binning하여 edge feature를 만든다.
        start_i, end_i = torch.nonzero(adj).T
        distance = torch.norm(pos[start_i] - pos[end_i], dim=1).unsqueeze(1)
        sigma_r = 1.5 ** torch.arange(self.args.distance_bins, device=distance.device) # 거리 thereshold
        edge_features = torch.exp(-(distance) ** 2 / 2 * (sigma_r) ** 2)
        return edge_features
        
        
    def convert_to_pyg_format(self, node_features, adjacency_matrices, node_pos):
        """
        Converts batched node features, adjacency matrices, and edge features to PyG format.
        
        Args:
        node_features (torch.Tensor): Shape (batch_size, num_nodes, num_node_features)
        adjacency_matrices (torch.Tensor): Shape (batch_size, num_nodes, num_nodes)
        node_pos (torch.Tensor) : Shape (batch_size, num_nodes, 3)
        
        Returns:
        torch_geometric.data.Batch: A batch of graphs in PyG format
        """
        batch_size = node_features.shape[0]
        pyg_data_list = []

        for i in range(batch_size):
            x = node_features[i]  # (num_nodes, num_node_features)
            adj = adjacency_matrices[i]  # (num_nodes, num_nodes)
            adj.fill_diagonal_(0)
            pos = node_pos[i]
            # Create edge_index from adjacency matrix
            edge_index = adj.nonzero().t().contiguous()
            edge_features = self.get_edge_features(adj, pos)
            
            # Create PyG Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_features)
            pyg_data_list.append(data)
        
        # Create a batch from the list of Data objects
        batch = Batch.from_data_list(pyg_data_list)
        
        return batch
    
    
    def get_intergraph_data_as_Batch(self, l_h_batch, t_h_batch, l_pos_batch, t_pos_batch, threshold = 12, num_bins=15):
        batch_size = l_pos_batch.shape[0]
        l_h_batch = l_h_batch.view(batch_size, -1, self.args.gnn_hidden_dim)
        t_h_batch = t_h_batch.view(batch_size, -1, self.args.gnn_hidden_dim)
        pyg_data_list = []
        
        for i in range(batch_size):
            l_h, t_h, l_pos, t_pos = l_h_batch[i], t_h_batch[i], l_pos_batch[i], t_pos_batch[i]
            # 1. get x
            x = torch.cat((l_h, t_h), dim=0)
            
            # 2. get edge index
            ligand_count = l_pos.shape[0]
            diff = l_pos.unsqueeze(1) - t_pos.unsqueeze(0)   
            distances = torch.sqrt((diff**2).sum(dim=2))
            valid_edges = distances <= threshold
            l_indices, t_indices = torch.where(valid_edges)
            t_indices += ligand_count # target indices 조정
            edge_index = torch.stack([
                torch.cat([l_indices, t_indices]),
                torch.cat([t_indices, l_indices])   
            ], dim=0)
            
            # 3. get edge_attr
            edge_dist = torch.cat([distances[valid_edges], distances[valid_edges]])
            sigma_r = 1.5 ** torch.arange(num_bins, device=edge_dist.device)
            diff = edge_dist.unsqueeze(1) - sigma_r.unsqueeze(0)
            closest_index = torch.abs(diff).argmin(dim=1)
            edge_attr = nn.functional.one_hot(closest_index, num_classes=num_bins).float()
        
            # Create PyG Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            pyg_data_list.append(data)
        
        batch = Batch.from_data_list(pyg_data_list)

        return batch


    def forward(self, sample, DM_min=0.5):
        # unpack sample
        (ligand_h,
         ligand_adj,
         target_h,
         target_adj,
         interaction_indice,
         ligand_pos,
         target_pos,
         rotor,
         ligand_vdw_radii,
         target_vdw_radii,
         ligand_valid,
         target_valid,
         ligand_non_metal,
         target_non_metal, _, _) = sample.values()
        
        # embed features    
        ligand_h = self.l_node_embedding(ligand_h)
        target_h = self.t_node_embedding(target_h)

        # get pyg format data
        ligand_batch = self.convert_to_pyg_format(ligand_h, ligand_adj, ligand_pos)
        target_batch = self.convert_to_pyg_format(target_h, target_adj, target_pos)
        
        # GNN forward propagation
        if self.args.gnn_type == 'GCN':
            ligand_h = self.l_gconv(x=ligand_batch.x, edge_index=ligand_batch.edge_index, batch=ligand_batch.batch)
            target_h = self.t_gconv(x=target_batch.x, edge_index=target_batch.edge_index, batch=target_batch.batch)
                
        # AttentiveFP forward propagation
        elif self.args.gnn_type == "AttentiveFP":
            ligand_h = self.l_attentiveFP(x=ligand_batch.x,
                                        edge_index=ligand_batch.edge_index,
                                        edge_attr=ligand_batch.edge_attr,
                                        batch=ligand_batch.batch)
            target_h = self.t_attentiveFP(x=target_batch.x,
                                        edge_index=target_batch.edge_index,
                                        edge_attr=target_batch.edge_attr,
                                        batch=target_batch.batch)
        
        # make interacting edge & pooling
        interacting_Batch = self.get_intergraph_data_as_Batch(ligand_h, target_h, ligand_pos, target_pos, threshold=12, num_bins=self.args.distance_bins)

        int_x, int_edge_index, int_edge_attr, int_batch = interacting_Batch.x,\
                                                          interacting_Batch.edge_index,\
                                                          interacting_Batch.edge_attr,\
                                                          interacting_Batch.batch # unpack Batch
        prj_edge_attr = self.int_projection_layer(int_edge_attr)
        int_feature = torch.cat((prj_edge_attr, int_x[int_edge_index[0]], int_x[int_edge_index[1]]), dim=1)
        int_feature = self.int_mlp(int_feature)
        # attention & pooling
        edge_batch = int_batch[int_edge_index[0]] # 몇 번째 edge가 몇 번째 sample에 속하는가 -> scatter 연산의 지표
        att_score = torch.tanh(self.w_h(int_feature))
        att_value = att_score * int_feature
        osum = scatter_sum(att_value, edge_batch, dim=0)
        omax, _ = scatter_max(int_feature, edge_batch, dim=0)
        o = torch.cat((osum, omax), dim=1)
        
        # predictor
        x = self.mlp(o)
        return x