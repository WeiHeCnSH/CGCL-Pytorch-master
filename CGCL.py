import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

import faiss
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class CGCL(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(CGCL, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type: the embedding size of the base model
        self.n_layers = config['n_layers']  # int type: the layer num of the base model
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization

        self.ssl_temp = config['ssl_temp']
        self.ssl_reg = config['ssl_reg']

        self.alpha = config['alpha']
        self.beta = config['beta']
        self.gamma = config['gamma']

        self.ssl_reg_alpha = config['ssl_reg_alpha']
        self.ssl_reg_beta = config['ssl_reg_beta']
        self.ssl_reg_gamma = config['ssl_reg_gamma']

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        self.norm_adj_mat = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        self.diag = torch.from_numpy(diag).to(self.device)
        D = sp.diags(diag)
        L = D @ A @ D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_mat, all_embeddings)
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list[:self.n_layers + 1], dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings, embeddings_list

    # loss 1
    def ssl_layer_loss(self, current_embedding, previous_embedding, user, item):
        # previous_embedding是中心的嵌入
        current_user_embeddings, current_item_embeddings = torch.split(current_embedding, [self.n_users, self.n_items])
        previous_user_embeddings_all, previous_item_embeddings_all = torch.split(previous_embedding,
                                                                                 [self.n_users, self.n_items])

        current_user_embeddings = current_user_embeddings[user]
        previous_user_embeddings = previous_user_embeddings_all[user]
        norm_user_emb1 = F.normalize(current_user_embeddings)
        norm_user_emb2 = F.normalize(previous_user_embeddings)
        norm_all_user_emb = F.normalize(previous_user_embeddings_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()
        # ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).mean()

        current_item_embeddings = current_item_embeddings[item]
        previous_item_embeddings = previous_item_embeddings_all[item]
        norm_item_emb1 = F.normalize(current_item_embeddings)
        norm_item_emb2 = F.normalize(previous_item_embeddings)
        norm_all_item_emb = F.normalize(previous_item_embeddings_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()
        # ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).mean()

        ssl_loss = self.ssl_reg_alpha * (self.alpha * ssl_loss_user + (1 - self.alpha) * ssl_loss_item)
        return ssl_loss

    # loss 2
    def ssl_canditation_layer_loss(self, current_embedding, previous_embedding, user, item):
        # previous_embedding是中心的嵌入
        layer_user_embeddings, layer_item_embeddings = torch.split(current_embedding, [self.n_users, self.n_items])
        previous_user_embeddings_all, previous_item_embeddings_all = torch.split(previous_embedding,
                                                                                 [self.n_users, self.n_items])

        previous_user_embeddings = previous_user_embeddings_all[user]
        current_user_embeddings = layer_item_embeddings[item]

        # user tower
        norm_user_emb1 = F.normalize(current_user_embeddings)
        norm_user_emb2 = F.normalize(previous_user_embeddings)
        norm_all_user_emb = F.normalize(previous_user_embeddings_all)

        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()
        # ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).mean()

        # item tower
        previous_item_embeddings = previous_item_embeddings_all[item]
        current_item_embeddings = layer_user_embeddings[user]

        norm_item_emb1 = F.normalize(current_item_embeddings)
        norm_item_emb2 = F.normalize(previous_item_embeddings)
        norm_all_item_emb = F.normalize(previous_item_embeddings_all)

        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()
        # ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).mean()

        ssl_loss = self.ssl_reg_beta * (self.beta * ssl_loss_user + (1 - self.beta) * ssl_loss_item)
        return ssl_loss

    # loss 3
    def calcuate_struct_loss(self, neighbor_embedding, center_embedding, user, item):
        neighbor_user_embedding, neighbor_item_embedding = torch.split(neighbor_embedding, [self.n_users, self.n_items])
        center_user_embedding, center_item_embedding = torch.split(center_embedding,
                                                                   [self.n_users, self.n_items])

        # 用户侧 注意锚点
        cent_user_embedding = center_user_embedding[user]  # l1
        neigh_item_embedding = neighbor_item_embedding[item]

        neigh_item_embedding = F.normalize(neigh_item_embedding)
        cent_user_embedding = F.normalize(cent_user_embedding)
        center_user_embedding = F.normalize(center_user_embedding)

        pos_score_user = torch.mul(neigh_item_embedding, cent_user_embedding).sum(dim=1)
        ttl_score_user = torch.matmul(neigh_item_embedding, center_user_embedding.transpose(0, 1))

        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        # 项目作为锚点
        neigh_user_embedding = neighbor_user_embedding[user]  # l2
        cent_item_embedding = center_item_embedding[item]

        neigh_user_embedding = F.normalize(neigh_user_embedding)
        cent_item_embedding = F.normalize(cent_item_embedding)
        center_item_embedding = F.normalize(center_item_embedding)

        pos_score_item = torch.mul(neigh_user_embedding, cent_item_embedding).sum(dim=1)
        ttl_score_item = torch.matmul(neigh_user_embedding, center_item_embedding.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()
        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.ssl_reg_gamma * (self.gamma * ssl_loss_user + (1 - self.gamma) * ssl_loss_item)
        return ssl_loss

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        center_embedding = embeddings_list[0]
        can_embedding = embeddings_list[1]
        context_embedding = embeddings_list[2]
        if self.ssl_reg_alpha > 1e-20:
            layer_loss = self.ssl_layer_loss(context_embedding, center_embedding, user, pos_item)
        else:
            layer_loss = torch.tensor(0.0)
        if self.ssl_reg_beta > 1e-20:
            can_loss = self.ssl_canditation_layer_loss(can_embedding, center_embedding, user, pos_item)
        else:
            can_loss = torch.tensor(0.0)
        if self.ssl_reg_gamma > 1e-20:
            str_loss = self.calcuate_struct_loss(context_embedding, can_embedding, user, pos_item)
        else:
            str_loss = torch.tensor(0.0)

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

        return mf_loss, self.reg_weight * reg_loss, layer_loss, can_loss, str_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, embedding_list = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
