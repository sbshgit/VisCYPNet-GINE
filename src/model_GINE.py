
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, BatchNorm1d, Dropout, Sequential
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool

# small helper to build 2-layer MLPs used inside GINEConv and head
def build_mlp(in_dim, hidden_dim, out_dim, activation=ReLU, use_batchnorm=False):
    layers = []
    layers.append(Linear(in_dim, hidden_dim))
    layers.append(activation())
    if use_batchnorm:
        layers.append(BatchNorm1d(hidden_dim))
    layers.append(Linear(hidden_dim, out_dim))
    return Sequential(*layers)

class GINEModel(nn.Module):
    """
    GINE-based graph classifier for molecular graphs.

    Forward signature:
        forward(x, edge_index, edge_attr=None, batch=None)

    Args:
        in_node_feats (int): dimensionality of node features.
        in_edge_feats (int): dimensionality of edge features (bond descriptors).
        model_params (dict): keys used:
            - model_embedding_size (int): hidden size for convs (default 64)
            - model_layers (int): number of GINE layers (default 4)
            - model_dropout_rate (float): dropout after dense layers (default 0.2)
            - model_dense_neurons (int): width of classification head (default 256)
            - use_bn (bool): whether to use BatchNorm after MLPs (default True)
    Returns:
        logits tensor of shape [batch_size] (raw logits, no sigmoid)
    """
    def __init__(self, in_node_feats: int, in_edge_feats: int, model_params: dict):
        super().__init__()
        # Hyperparams / sensible defaults
        self.embedding_size = model_params.get("model_embedding_size", 64)
        self.n_layers = model_params.get("model_layers", 4)
        self.dropout_rate = float(model_params.get("model_dropout_rate", 0.2))
        self.dense_neurons = int(model_params.get("model_dense_neurons", 256))
        self.use_bn = bool(model_params.get("use_bn", True))

        # 1) Encoders: map raw node/edge features → embedding_size
        #    This ensures node and edge tensors have compatible dimensions.
        self.node_encoder = Linear(in_node_feats, self.embedding_size)
        # Edge encoder maps bond descriptors -> embedding_size. If edge features are None,
        # this will simply not be used.
        self.edge_encoder = Linear(in_edge_feats, self.embedding_size) if in_edge_feats is not None else None

        # 2) Build stacked GINEConv layers
        self.convs = ModuleList()
        self.bns = ModuleList() if self.use_bn else None

        # Each GINEConv needs an `nn` (MLP) that accepts embedding_size and outputs embedding_size.
        for i in range(self.n_layers):
            mlp = build_mlp(self.embedding_size, self.embedding_size, self.embedding_size,
                            activation=ReLU, use_batchnorm=False)
            # GINEConv will internally add neighbor node features + edge_attr (assumes same dim).
            conv = GINEConv(mlp)
            self.convs.append(conv)
            if self.use_bn:
                self.bns.append(BatchNorm1d(self.embedding_size))

        # 3) Classification head
        #    We pool node representations with (sum + mean) concatenation
        head_in = self.embedding_size * 2  # sum + mean
        self.head = nn.Sequential(
            Linear(head_in, self.dense_neurons),
            ReLU(),
            Dropout(self.dropout_rate),
            Linear(self.dense_neurons, max(8, self.dense_neurons // 2)),
            ReLU(),
            Dropout(self.dropout_rate),
            Linear(max(8, self.dense_neurons // 2), 1)  # single logit
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        x: [num_nodes, in_node_feats]
        edge_index: [2, num_edges] (long tensor)
        edge_attr: [num_edges, in_edge_feats] or None
        batch: [num_nodes] assignment tensor that maps nodes -> graph index
        """
        # If `batch` not provided assume single graph (all nodes belong to batch 0)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        # Encode node/edge features
        x = self.node_encoder(x)          # -> [num_nodes, embedding_size]
        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)  # -> [num_edges, embedding_size]

        # Stacked GINE layers
        for i, conv in enumerate(self.convs):
            # GINEConv signature: conv(x, edge_index, edge_attr)
            x = conv(x, edge_index, edge_attr)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)

        # Pool: sum (add) + mean concatenation
        sum_pool = global_add_pool(x, batch)   # [batch_size, embedding_size]
        mean_pool = global_mean_pool(x, batch) # [batch_size, embedding_size]
        graph_repr = torch.cat([sum_pool, mean_pool], dim=1)  # [batch_size, embedding_size*2]

        # Head → single logit per graph
        logits = self.head(graph_repr).view(-1)  # [batch_size]
        return logits
