import torch.nn as nn
import torch as th
import dgl.function as fn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch.conv import  RelGraphConv
import dgl
from  AGNN.utils import *


class MyGraphConv(nn.Module):
    """Graph convolution module used in the my model.

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    dropout : torch.nn.Dropout, optional
        Optional external dropout layer.

    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 dropout_rate=0.0):
        super(MyGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.dropout = nn.Dropout(dropout_rate)
        self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, feat):
        """Compute graph convolution.
        Normalizer constant :math:`c_{ij}` is stored as two node data "ci"
        and "cj".

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature
        weight : torch.Tensor, optional
            Optional external weight tensor.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat, feat_dst = feat      # dst feature not used
            cj = graph.srcdata['cj']
            ci = graph.dstdata['ci']
            feat=dot_or_identity(feat,self.weight)

            feat = feat * self.dropout(cj)
            graph.srcdata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            rst = rst * ci
        return rst

class MyLayer(nn.Module):
    def __init__(self,
                 user_in_units,
                 movie_in_units,
                 msg_units,
                 out_units,
                 rating_vals,
                 dropout_rate=0.0,
                 agg='stack',  # or 'sum'
                 agg_act=None,
                 out_act=None
                ):
        super(MyLayer, self).__init__()
        len_rate = len(rating_vals)
        self.ufc = nn.Linear(msg_units*len_rate, out_units)
        self.ifc = nn.Linear(msg_units*len_rate, out_units)
        # if agg=='stack':
        #     msg_units = msg_units // len(rating_vals)
        self.dropout = nn.Dropout(dropout_rate)
        # self.W_r = nn.ParameterDict()
        self.agg=agg
        subConv = {}
        for rating in rating_vals:
            rating=str(rating)
            subConv[(rating).replace('.','_')] = MyGraphConv(user_in_units,
                                            msg_units,
                                            dropout_rate=dropout_rate)
            subConv[(rating+'ed').replace('.','_')] = MyGraphConv(movie_in_units,
                                                msg_units,
                                                dropout_rate=dropout_rate)
        # subConv['trust'] = MyGraphConv(movie_in_units,
        #                                     msg_units,
        #                                     dropout_rate=dropout_rate)
        self.conv = dglnn.HeteroGraphConv(subConv,agg)
        self.agg_act = get_activation(agg_act)
        self.out_act = get_activation(out_act)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph,in_feats):
        """Forward function
        Parameters
        ----------
        graph : DGLHeteroGraph
            User-movie rating graph. It should contain two node types: "user"
            and "movie" and many edge types each for one rating value.
        ufeat : torch.Tensor, optional
            User features. If None, using an identity matrix.
        ifeat : torch.Tensor, optional
            Movie features. If None, using an identity matrix.

        Returns
        -------
        new_ufeat : torch.Tensor
            New user features
        new_ifeat : torch.Tensor
            New movie features
        """
        # in_feats = {'user' : ufeat, 'movie' : ifeat}
        out_feats = self.conv(graph, (in_feats,in_feats))
        ufeat = out_feats['user']
        ifeat = out_feats['item']
        ufeat = ufeat.view(ufeat.shape[0], -1)
        ifeat = ifeat.view(ifeat.shape[0], -1)
        # fc and non-linear
        ufeat = self.agg_act(ufeat)
        ifeat = self.agg_act(ifeat)
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        ufeat = self.ufc(ufeat)
        ifeat = self.ifc(ifeat)
        return {'user':self.out_act(ufeat), 'item':self.out_act(ifeat)}

class MLPPredictor(nn.Module):
    def __init__(self,in_features,dropout_rate=0.5):
        super().__init__()
        self.W=nn.Sequential(nn.Linear(2* in_features,in_features),
                            nn.ReLU(),
                            nn.Linear(in_features, 1),
                            nn.Sigmoid()
                            )
        self.dropout = nn.Dropout(dropout_rate)
    def apply_edges(self,edges):
        return {'score':self.W(th.cat((edges.src['h'],edges.dst['h']),1))}

    def forward(self,g: dgl.DGLHeteroGraph,h,etype='interact'):
        with g.local_scope():
            g.nodes['user'].data['h']=self.dropout(h['user'])
            g.nodes['item'].data['h'] = self.dropout(h['item'])
            g.apply_edges(self.apply_edges,etype=etype)
            return g.edges[etype].data['score']*5

def dot_or_identity(A, B, device=None):
    # if A is None, treat as identity matrix
    if A is None:
        return B
    elif len(A.shape) == 1:
        if device is None:
            return B[A]
        else:
            return B[A].to(device)
    else:
        return A @ B






