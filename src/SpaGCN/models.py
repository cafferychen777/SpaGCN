import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sklearn.cluster import KMeans
import torch.optim as optim
import numpy as np
from .layers import GraphConvolution
from ._clustering import graph_cluster_labels


def _student_t_assignments(embeddings, cluster_centers, alpha):
    """Compute normalized DEC soft assignments with a Student's t kernel."""
    if alpha <= 0:
        raise ValueError("alpha must be greater than zero")

    squared_distances = torch.sum(
        (embeddings.unsqueeze(1) - cluster_centers) ** 2,
        dim=2,
    )
    unnormalized = torch.pow(
        1.0 + squared_distances / alpha,
        -((alpha + 1.0) / 2.0),
    )
    return unnormalized / torch.sum(unnormalized, dim=1, keepdim=True)


def _float_tensor(value):
    return torch.as_tensor(value, dtype=torch.float32)


def _cluster_centers(features, labels):
    labels = np.asarray(labels)
    _, inverse = np.unique(labels, return_inverse=True)
    centers = np.zeros((inverse.max() + 1, features.shape[1]), dtype=np.float32)
    np.add.at(centers, inverse, features)
    centers /= np.bincount(inverse)[:, None]
    return centers


def _make_optimizer(parameters, opt, lr, weight_decay):
    if opt == "sgd":
        return optim.SGD(parameters, lr=lr, momentum=0.9)
    if opt in {"adam", "admin"}:
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError("opt must be 'sgd' or 'adam'")


class simple_GC_DEC(nn.Module):
    def __init__(self, nfeat, nhid, alpha=0.2):
        super(simple_GC_DEC, self).__init__()
        self.gc = GraphConvolution(nfeat, nhid)
        self.nhid = nhid
        # self.mu determined by the init method
        self.alpha = alpha

    def forward(self, x, adj):
        x = self.gc(x, adj)
        q = _student_t_assignments(x, self.mu, self.alpha)
        return x, q

    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(
                torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1)
            )

        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        # weight = q ** 2 / q.sum(0)
        # return torch.transpose((torch.transpose(weight,0,1) / weight.sum(1)),0,1)e
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def fit(
        self,
        X,
        adj,
        lr=0.001,
        max_epochs=5000,
        update_interval=3,
        trajectory_interval=50,
        weight_decay=5e-4,
        opt="sgd",
        init="leiden",
        n_neighbors=10,
        res=0.4,
        n_clusters=10,
        init_spa=True,
        tol=1e-3,
    ):
        self.trajectory = []
        X = _float_tensor(X)
        adj = _float_tensor(adj)
        with torch.no_grad():
            features = self.gc(X, adj)
        # ----------------------------------------------------------------
        if init == "kmeans":
            print("Initializing cluster centers with kmeans, n_clusters known")
            self.n_clusters = n_clusters
            kmeans = KMeans(self.n_clusters, n_init=20)
            if init_spa:
                # ------Kmeans use exp and spatial
                y_pred = kmeans.fit_predict(features.detach().numpy())
            else:
                # ------Kmeans only use exp info, no spatial
                y_pred = kmeans.fit_predict(X.numpy())
        elif init in {"leiden", "louvain"}:
            print("Initializing cluster centers with Leiden, resolution = ", res)
            values = features.numpy() if init_spa else X.numpy()
            y_pred = graph_cluster_labels(
                values,
                method=init,
                n_neighbors=n_neighbors,
                resolution=res,
            )
            self.n_clusters = len(np.unique(y_pred))
        # ----------------------------------------------------------------
        y_pred_last = y_pred
        self.mu = Parameter(torch.Tensor(self.n_clusters, self.nhid))
        self.trajectory.append(y_pred)
        cluster_centers = _cluster_centers(features.numpy(), y_pred)
        with torch.no_grad():
            self.mu.copy_(_float_tensor(cluster_centers))
        optimizer = _make_optimizer(self.parameters(), opt, lr, weight_decay)
        self.train()
        for epoch in range(max_epochs):
            if epoch % update_interval == 0:
                with torch.no_grad():
                    _, q = self.forward(X, adj)
                    p = self.target_distribution(q)
            if epoch % 10 == 0:
                print("Epoch ", epoch)
            optimizer.zero_grad()
            z, q = self(X, adj)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()
            if epoch % trajectory_interval == 0:
                self.trajectory.append(torch.argmax(q, dim=1).detach().cpu().numpy())

            # Check stop criterion
            y_pred = torch.argmax(q, dim=1).detach().cpu().numpy()
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / X.shape[0]
            y_pred_last = y_pred
            if epoch > 0 and (epoch - 1) % update_interval == 0 and delta_label < tol:
                print("delta_label ", delta_label, "< tol ", tol)
                print("Reach tolerance threshold. Stopping training.")
                print("Total epoch:", epoch)
                break

    def fit_with_init(
        self,
        X,
        adj,
        init_y,
        lr=0.001,
        max_epochs=5000,
        update_interval=1,
        weight_decay=5e-4,
        opt="sgd",
    ):
        print("Initializing cluster centers with kmeans.")
        optimizer = _make_optimizer(self.parameters(), opt, lr, weight_decay)
        X = _float_tensor(X)
        adj = _float_tensor(adj)
        with torch.no_grad():
            features, _ = self.forward(X, adj)
        cluster_centers = _cluster_centers(features.detach().numpy(), init_y)
        with torch.no_grad():
            self.mu.copy_(_float_tensor(cluster_centers))
        self.train()
        for epoch in range(max_epochs):
            if epoch % update_interval == 0:
                with torch.no_grad():
                    _, q = self.forward(X, adj)
                    p = self.target_distribution(q)
            optimizer.zero_grad()
            z, q = self(X, adj)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()

    def predict(self, X, adj):
        with torch.no_grad():
            z, q = self(_float_tensor(X), _float_tensor(adj))
        return z, q


class GC_DEC(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, n_clusters=None, dropout=0.5, alpha=0.2):
        super(GC_DEC, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.dropout = dropout
        self.mu = Parameter(torch.Tensor(n_clusters, nhid2))
        self.n_clusters = n_clusters
        self.alpha = alpha

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=True)
        x = self.gc2(x, adj)
        q = _student_t_assignments(x, self.mu, self.alpha)
        return x, q

    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(
                torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1)
            )

        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        # weight = q ** 2 / q.sum(0)
        # return torch.transpose((torch.transpose(weight,0,1) / weight.sum(1)),0,1)e
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def fit(
        self,
        X,
        adj,
        lr=0.001,
        max_epochs=10,
        update_interval=5,
        weight_decay=5e-4,
        opt="sgd",
        init="leiden",
        n_neighbors=10,
        res=0.4,
    ):
        self.trajectory = []
        print("Initializing cluster centers with kmeans.")
        optimizer = _make_optimizer(self.parameters(), opt, lr, weight_decay)

        X = _float_tensor(X)
        adj = _float_tensor(adj)
        with torch.no_grad():
            features, _ = self.forward(X, adj)
        # ----------------------------------------------------------------

        if init == "kmeans":
            # Kmeans only use exp info, no spatial
            # kmeans = KMeans(self.n_clusters, n_init=20)
            # y_pred = kmeans.fit_predict(X)  #Here we use X as numpy
            # Kmeans use exp and spatial
            kmeans = KMeans(self.n_clusters, n_init=20)
            y_pred = kmeans.fit_predict(features.numpy())
        elif init in {"leiden", "louvain"}:
            y_pred = graph_cluster_labels(
                features.numpy(),
                method=init,
                n_neighbors=n_neighbors,
                resolution=res,
            )
        # ----------------------------------------------------------------
        self.trajectory.append(y_pred)
        cluster_centers = _cluster_centers(features.numpy(), y_pred)
        with torch.no_grad():
            self.mu.copy_(_float_tensor(cluster_centers))
        self.train()
        for epoch in range(max_epochs):
            if epoch % update_interval == 0:
                with torch.no_grad():
                    _, q = self.forward(X, adj)
                    p = self.target_distribution(q)
            if epoch % 100 == 0:
                print("Epoch ", epoch)
            optimizer.zero_grad()
            z, q = self(X, adj)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()
            self.trajectory.append(torch.argmax(q, dim=1).detach().cpu().numpy())

    def fit_with_init(
        self,
        X,
        adj,
        init_y,
        lr=0.001,
        max_epochs=10,
        update_interval=1,
        weight_decay=5e-4,
        opt="sgd",
    ):
        print("Initializing cluster centers with kmeans.")
        optimizer = _make_optimizer(self.parameters(), opt, lr, weight_decay)
        X = _float_tensor(X)
        adj = _float_tensor(adj)
        with torch.no_grad():
            features, _ = self.forward(X, adj)
        cluster_centers = _cluster_centers(features.numpy(), init_y)
        with torch.no_grad():
            self.mu.copy_(_float_tensor(cluster_centers))
        self.train()
        for epoch in range(max_epochs):
            if epoch % update_interval == 0:
                with torch.no_grad():
                    _, q = self.forward(X, adj)
                    p = self.target_distribution(q)
            optimizer.zero_grad()
            z, q = self(X, adj)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()

    def predict(self, X, adj):
        with torch.no_grad():
            z, q = self(_float_tensor(X), _float_tensor(adj))
        return z, q
