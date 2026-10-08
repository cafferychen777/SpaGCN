# cd Desktop/GCN/pygcn/env2/
import numpy as np
from anndata import AnnData
import torch
from ._compat import SPAGCN_1_2_7, resolve_compat
from ._data import embed_expression, exp_adjacency
from .models import simple_GC_DEC


class SpaGCN(object):
    """SpaGCN spatial domain model.

    Every ``train`` call accepts ``compat``:

    - ``"spagcn-1.2.7"`` (default) reproduces the published SpaGCN 1.2.7
      implementation, including fixed cluster centers, its soft-assignment
      expression, Louvain initialization, and float64 preprocessing.
    - ``"modern"`` trains the cluster centers, uses the DEC Student's t kernel,
      initializes with Leiden, and uses the float32/sparse preprocessing of
      spagcn-modern 1.3. The upstream hyperparameters were tuned for the
      original behavior, so results differ.
    """

    def __init__(self):
        super(SpaGCN, self).__init__()
        self.l = None

    def set_l(self, l):
        self.l = l

    def train(
        self,
        adata,
        adj,
        num_pcs=50,
        lr=0.005,
        max_epochs=2000,
        weight_decay=0,
        opt="admin",
        init_spa=True,
        init=None,  # louvain, leiden, or kmeans; None follows compat
        n_neighbors=10,  # for louvain/leiden
        n_clusters=None,  # for kmeans
        res=0.4,  # for louvain/leiden
        tol=1e-3,
        compat=None,
    ):
        compat = resolve_compat(compat)
        if self.l is None:
            raise ValueError("l must be set before fitting the model!")
        if adata.shape[0] != adj.shape[0] or adj.shape[0] != adj.shape[1]:
            raise ValueError("adata and adjacency dimensions must agree")

        embed = embed_expression(adata.X, num_pcs, compat)
        adj_exp = exp_adjacency(adj, self.l, compat)
        self._train_prepared(
            embed,
            adj_exp,
            num_pcs=num_pcs,
            lr=lr,
            max_epochs=max_epochs,
            weight_decay=weight_decay,
            opt=opt,
            init_spa=init_spa,
            init=init,
            n_neighbors=n_neighbors,
            n_clusters=n_clusters,
            res=res,
            tol=tol,
            compat=compat,
        )

    def _train_prepared(
        self,
        embed,
        adj_exp,
        num_pcs=50,
        lr=0.005,
        max_epochs=2000,
        weight_decay=0,
        opt="admin",
        init_spa=True,
        init=None,
        n_neighbors=10,
        n_clusters=None,
        res=0.4,
        tol=1e-3,
        compat=None,
    ):
        self.compat = resolve_compat(compat)
        self.num_pcs = num_pcs
        self.res = res
        self.lr = lr
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.opt = opt
        self.init_spa = init_spa
        self.init = init
        self.n_neighbors = n_neighbors
        self.n_clusters = n_clusters
        self.tol = tol
        self.model = simple_GC_DEC(embed.shape[1], embed.shape[1], compat=self.compat)
        self.model.fit(
            embed,
            adj_exp,
            lr=self.lr,
            max_epochs=self.max_epochs,
            weight_decay=self.weight_decay,
            opt=self.opt,
            init_spa=self.init_spa,
            init=self.init,
            n_neighbors=self.n_neighbors,
            n_clusters=self.n_clusters,
            res=self.res,
            tol=self.tol,
        )
        self.embed = embed
        self.adj_exp = adj_exp

    def predict(self):
        z, q = self.model.predict(self.embed, self.adj_exp)
        y_pred = torch.argmax(q, dim=1).cpu().numpy()
        prob = q.cpu().numpy()
        return y_pred, prob


class multiSpaGCN(object):
    def __init__(self):
        super(multiSpaGCN, self).__init__()
        self.l = None

    def train(
        self,
        adata_list,
        adj_list,
        l_list,
        num_pcs=50,
        lr=0.005,
        max_epochs=2000,
        weight_decay=0,
        opt="admin",
        init_spa=True,
        init=None,  # louvain, leiden, or kmeans; None follows compat
        n_neighbors=10,  # for louvain/leiden
        n_clusters=None,  # for kmeans
        res=0.4,  # for louvain/leiden
        tol=1e-3,
        compat=None,
    ):
        self.compat = compat = resolve_compat(compat)
        self.num_pcs = num_pcs
        self.res = res
        self.lr = lr
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.opt = opt
        self.init_spa = init_spa
        self.init = init
        self.n_neighbors = n_neighbors
        self.n_clusters = n_clusters
        self.res = res
        self.tol = tol
        num_spots = 0
        for i in adata_list:
            num_spots += i.shape[0]
        # SpaGCN 1.2.7 used np.empty here, leaving the cross-sample blocks
        # uninitialized; both modes use zeros (no cross-sample edges).
        adj_dtype = np.float64 if compat == SPAGCN_1_2_7 else np.float32
        adj_exp_all = np.zeros((num_spots, num_spots), dtype=adj_dtype)
        start = 0
        for i in range(len(l_list)):
            length_scale = l_list[i]
            adj = adj_list[i]
            adj_exp = exp_adjacency(adj, length_scale, compat)
            adj_exp_all[
                start : start + adj_exp.shape[0], start : start + adj_exp.shape[0]
            ] = adj_exp
            start += adj_exp.shape[0]
        batch_cat = [str(i) for i in range(len(l_list))]
        self.adata_all = AnnData.concatenate(
            *adata_list,
            join="inner",
            batch_key="dataset_batch",
            batch_categories=batch_cat,
        )
        embed = embed_expression(self.adata_all.X, self.num_pcs, compat)
        # ----------Train model----------
        self.model = simple_GC_DEC(embed.shape[1], embed.shape[1], compat=compat)
        self.model.fit(
            embed,
            adj_exp_all,
            lr=self.lr,
            max_epochs=self.max_epochs,
            weight_decay=self.weight_decay,
            opt=self.opt,
            init_spa=self.init_spa,
            init=self.init,
            n_neighbors=self.n_neighbors,
            n_clusters=self.n_clusters,
            res=self.res,
            tol=self.tol,
        )
        self.embed = embed
        self.adj_exp = adj_exp_all

    def predict(self):
        z, q = self.model.predict(self.embed, self.adj_exp)
        y_pred = torch.argmax(q, dim=1).cpu().numpy()
        prob = q.cpu().numpy()
        return y_pred, prob
