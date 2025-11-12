import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import math
from torch.testing._internal.common_subclass import SparseTensor
from utils.user_att_count import get_item_att
from utils.build_train_rec_data_loader_new import pad_list_of_list
from sentence_transformers import SentenceTransformer
import random
from torch.autograd import Variable
from torch import LongTensor
from torch import FloatTensor
from numpy.random import RandomState
import time
from sklearn.manifold import TSNE
import os
class MyRec(nn.Module):
    def __init__(self, config,
                 vocab_size, user_size, product_size,
                 query_max_length, word_dists, product_dists,
                 aspect_dists, value_dists,
                 av_id2word,
                 aword_idx2_aid,
                 aspect_keys, value_keys,
                 aspect_value_count_dic, 
                 window_size, embedding_size, query_weight, qnet_struct,
                 comb_net_struct,
                 model_net_struct, 
                 similarity_func, negative_sample,
                 av_vocab_size,
                 group_product_count=1000,
                 value_loss_func='softmax',
                 loss_ablation='',
                 scale_grad=False,
                 is_emb_sparse=False,
                 is_av_embed_shared=False,
                 aspect_prob_type='softmax',
                 likelihood_way='av', 
                 use_neg_sample_per_aspect=False, personalized=False):
        super().__init__()
        self.layer_aggregate = config.layer_aggregate
        self.gpu = False
        self.hidden_dim = config.hidden_dim
        self.user_num = config.user_num
        self.item_num = config.item_num
        self.n_layers = config.nlayer
        self.attribute_num = config.attribute_num
        self.user_index = torch.tensor([_ for _ in range(self.user_num)])
        self.item_index = torch.tensor([_ for _ in range(self.item_num)])
        self.attribute_index = torch.tensor([_ for _ in range(self.attribute_num)])
        self.attribute_index_convps = torch.tensor([_ for _ in range(self.attribute_num + 1)])
        self.user_graph_index = torch.tensor([_ for _ in range(self.user_num)])
        self.item_graph_index = torch.tensor([_ for _ in range(self.user_num, self.user_num + self.item_num + 1)])
        self.attribute_graph_index = torch.tensor(
            [_ for _ in range(self.user_num + self.item_num, self.user_num + self.item_num + self.attribute_num + 1)])
        self.gcs = nn.ModuleList()
        self.softmax = nn.Softmax(dim=-1)
        self.nlayer = config.nlayer
        self.conv_name = config.conv_name
        self.n_heads = config.n_heads
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.tanh = nn.Tanh()
        self.graph_rep = None
        self.eps = torch.tensor(1e-9)
        self.drop = nn.Dropout(config.drop)
        if self.layer_aggregate == 'last_layer' or self.layer_aggregate == 'mean':
            self.graph_dim = self.hidden_dim
        elif self.layer_aggregate == 'concat':
            self.graph_dim = (self.n_layers + 1) * self.hidden_dim
        else:
            print("not support layer_aggregate type : {} !!!".format(self.layer_aggregate))
        self.graph_layer_rep = None
        self.current_user = None
        self.neg_user = None
        data_path = './data/TeCQR/Graph_generate_data'
        num_query = {}
        with open('%s/question_tag.txt' % data_path, 'r', encoding='UTF-8') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    num_query[int(parts[0])] = parts[1]
        sbert = SentenceTransformer("/home/xxxxxx/TeCQR/prelm/all-MiniLM-L6-v2")
        with open('%s/question_new_num.txt' % data_path, 'r', encoding='UTF-8') as file:
            self.old_new = {}
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    self.old_new[int(parts[0])] = int(parts[1])
        self.all_query = {}
        self.all_query_new = {}
        self.train_test = False
        if self.train_test == True:
            with open('%s/all_pro.json' % data_path, 'r') as f:
                f_data = json.load(f)
                for pid in f_data.keys():
                    self.all_query[self.old_new[int(pid)]] = torch.Tensor(sbert.encode(num_query[self.old_new[(int(pid))]]))
                    self.all_query_new[int(pid)] = self.all_query[self.old_new[int(pid)]]
            self.prodcut_emb_list = []
            for pid in range(len(self.all_query_new)):
                self.prodcut_emb_list.append(self.all_query_new[pid])
            self.product_tensor = torch.stack(self.prodcut_emb_list)
        print("products are loaded\n")
        if self.gpu:
            self.item_index = self.item_index.cuda()
            self.attribute_index = self.attribute_index.cuda()
            self.attribute_index_convps = self.attribute_index_convps.cuda()
            self.item_graph_index = self.item_graph_index.cuda()
            self.attribute_graph_index = self.attribute_graph_index.cuda()
            self.eps = self.eps.cuda()
            self.drop = self.drop.cuda()
            self.user_index = self.user_index.cuda()
            self.item_index = self.item_index.cuda()
            self.attribute_index = self.attribute_index.cuda()
            self.user_graph_index = self.user_graph_index.cuda()
            self.item_graph_index = self.item_graph_index.cuda()
            self.attribute_graph_index = self.attribute_graph_index.cuda()
        self.group_product_count = group_product_count
        self.debug = False
        self.aspect_prob_type = aspect_prob_type
        self.likelihood_way = likelihood_way
        self.vocab_size = vocab_size + 1
        self.av_vocab_size = av_vocab_size
        self.aword_idx2_aid = aword_idx2_aid
        self.loss_ablation = loss_ablation
        self.padding_idx = self.vocab_size - 1
        self.user_size = user_size
        self.product_size = product_size
        self.query_max_length = query_max_length
        self.qnet_struct = qnet_struct
        self.comb_net_struct = comb_net_struct
        self.prod_aspect_struct = comb_net_struct
        self.model_net_struct = model_net_struct
        self.similarity_func = similarity_func
        self.embedding_size = embedding_size
        self.is_av_embed_shared = is_av_embed_shared
        self.is_emb_sparse = is_emb_sparse
        self.use_neg_sample_per_aspect = use_neg_sample_per_aspect
        self.value_loss_func = value_loss_func
        self.softmax_func = nn.Softmax(dim=-1)  
        self.qlinear = nn.Linear(self.embedding_size, self.embedding_size) 
        if self.comb_net_struct == "linear_fc":
            self.word_av_fc = nn.Linear(2 * self.embedding_size, self.embedding_size)  
        self.word_emb = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.padding_idx,
                                      scale_grad_by_freq=scale_grad, sparse=self.is_emb_sparse)
        self.word_bias = nn.Parameter(torch.zeros(self.vocab_size), requires_grad=True)
        self.user_emb = nn.Embedding(self.user_size, self.embedding_size, sparse=self.is_emb_sparse)
        self.product_emb = nn.Embedding(self.product_size, self.embedding_size, sparse=self.is_emb_sparse)
        self.asp_weight_emb = nn.Embedding(self.av_vocab_size, 1, sparse=self.is_emb_sparse)
        self.query_weight_emb = nn.Embedding(self.product_size, 1, sparse=self.is_emb_sparse)
        self.product_bias = nn.Parameter(torch.zeros(self.product_size), requires_grad=True)
        self.n_negs = negative_sample
        self.aspect_keys = torch.LongTensor(aspect_keys)
        self.value_keys = torch.LongTensor(value_keys) 
        self.personalized = personalized
        self.aspect_padding_idx = len(self.aspect_keys) - 1
        self.value_padding_idx = len(self.value_keys) - 1
        self.aspect_bias = nn.Parameter(torch.zeros(len(self.aspect_keys)), requires_grad=True)
        self.words_dists = FloatTensor(word_dists)
        self.product_dists = FloatTensor(product_dists)
        self.aspect_dists = FloatTensor(aspect_dists)
        self.value_dists = FloatTensor(value_dists)
        if self.is_av_embed_shared:
            self.av_padding_idx = self.av_vocab_size - 1
            self.av_word_emb = nn.Embedding(self.av_vocab_size, self.embedding_size, padding_idx=self.av_padding_idx,
                                            scale_grad_by_freq=scale_grad)  
            self.value_bias = nn.Parameter(torch.zeros(self.av_vocab_size), requires_grad=True)
        else:
            self.av_padding_idx = self.av_vocab_size-1
            self.av_word_emb = nn.Embedding(self.av_vocab_size, self.embedding_size, padding_idx=self.av_padding_idx,
                                            scale_grad_by_freq=scale_grad) 
            self.value_bias = nn.Parameter(torch.zeros(self.av_vocab_size), requires_grad=True)
        if self.value_loss_func == 'sep_emb':
            self.neg_av_word_emb = nn.Embedding(self.av_vocab_size, self.embedding_size,
                                                padding_idx=self.av_padding_idx,
                                                scale_grad_by_freq=scale_grad) 
            self.neg_value_bias = nn.Parameter(torch.zeros(self.av_vocab_size), requires_grad=True)
        print("CHECK av_word_emb.size myrec: {}".format(self.av_word_emb.weight.size()))
        self.value_dists_per_aspect = dict()
        for aspect in aspect_value_count_dic:
            value_keys_for_a = aspect_value_count_dic[aspect].keys()
            value_dists_for_a = FloatTensor(
                self.neg_distributes([aspect_value_count_dic[aspect][v] for v in value_keys_for_a]))
            self.value_dists_per_aspect[self.aword_idx2_aid[aspect]] = (np.asarray(value_keys_for_a), value_dists_for_a)
        self.query_weight = query_weight
        self.als_param = nn.Parameter(torch.zeros(1))
        self.init_weights()
        self.prepare_qup_time = 0.
        self.up_loss_time = 0.
        self.uw_loss_time = 0.
        self.pw_loss_time = 0.
        self.nce_neg_gen_time = 0.
        self.nce_get_vec_time = 0.
        self.nce_compute_loss_time = 0.
        self.nce_unsqueeze_time = 0.
        self.nce_bmm_time = 0.
        self.group_tq_training= 0.
        self.group_iav_loss = 0.
        self.group_up_loss1 = 0.
        self.group_uw_loss = 0.
        self.group_pw_loss = 0.
        self.group_up_loss2 = 0.
        self.group_av_loss = 0.
        self.group_word_loss = 0.
        self.group_iav_pos_loss = 0.
        self.group_iav_neg_loss = 0.
        if self.gpu:
            self.qlinear = self.qlinear.cuda()
            self.word_emb = self.word_emb.cuda()
            self.user_emb = self.user_emb.cuda()
            self.product_emb = self.product_emb.cuda()
            self.aspect_keys = self.aspect_keys.cuda()
            self.value_keys = self.value_keys.cuda()
            self.words_dists = self.words_dists.cuda()
            self.product_dists = self.product_dists.cuda()
            self.aspect_dists = self.aspect_dists.cuda()
            self.value_dists = self.value_dists.cuda()
            self.av_word_emb = self.av_word_emb.cuda()
            self.asp_weight_emb = self.asp_weight_emb.cuda()
            self.query_weight_emb = self.query_weight_emb.cuda()
    def init_para(self):
        all_embed = nn.Parameter(
            torch.FloatTensor(self.user_num + self.item_num + self.attribute_num, self.hidden_dim)
        )
        nn.init.xavier_uniform_(all_embed)
        self.user_embed.weight.data = all_embed[:self.user_num].data
        self.item_embed.weight.data = all_embed[self.user_num:self.user_num + self.item_num].data
        self.attribute_embed.weight.data = all_embed[
                                           self.user_num + self.item_num:self.user_num + self.item_num + self.attribute_num].data
    def init_global_aspect_emb(self):
        if self.word_emb.weight.is_cuda:
            self.aspect_keys = self.aspect_keys.cuda()
            self.value_keys = self.value_keys.cuda()
            self.value_dists = self.value_dists.cuda()
            self.words_dists = self.words_dists.cuda()
            self.aspect_dists = self.aspect_dists.cuda()
            self.product_dists = self.product_dists.cuda()
    def init_aspect_vecs_for_test(self):
        self.aspect_emb = self.get_embedding_from_words_asp(
            self.av_word_emb, self.aspect_keys, self.av_padding_idx)
        self.aspect_size = len(self.aspect_keys)
    def neg_distributes(self, weights, distortion=0.75):
        weights = np.asarray(weights)
        wf = weights / weights.sum()
        wf = np.power(wf, distortion)
        wf = wf / wf.sum()
        return wf
    def init_weights(self):
        self.user_emb.weight.data.uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)
        self.product_emb.weight.data.uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)
        num_embeddings = self.av_vocab_size
        embedding_dim = 1
        first_half = torch.rand(num_embeddings - num_embeddings // 2, embedding_dim) * 0.5
        second_half = torch.rand(num_embeddings - num_embeddings // 2, embedding_dim) * 0.5
        weights = torch.cat((first_half, second_half), dim=0)
        self.asp_weight_emb.weight.data.copy_(weights)
        self.query_weight_emb.weight.data.uniform_(1, 2)
        print("init_weights")
        if self.train_test == True:
            self.product_emb.weight.data.copy_(self.product_tensor)
        if self.value_loss_func == 'sep_emb':
            self.neg_av_word_emb.weight.data.uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)
            self.neg_av_word_emb.weight.data[self.av_padding_idx] = 0
        if not self.is_av_embed_shared:
            self.av_word_emb.weight.data.uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)
            self.av_word_emb.weight.data[self.av_padding_idx] = 0
    def forward(self, batch_data,prev_losses):
        loss = 0
        if self.model_net_struct == 'AVHEM':
            user_idxs, product_idxs, query_word_idxs, word_idxs, \
            wrapped_neg_idxs, aspect_value_entries = batch_data
            av_loss, ia_loss, qq_training, up_loss = self.qup_av_nce_loss(batch_data)
            word_list = [int(x) for x in word_idxs if x != 'None']
            batch_data = [user_idxs, product_idxs, query_word_idxs, word_list] + [wrapped_neg_idxs[:3]]
            loss = av_loss 
            self.group_av_loss += av_loss.item()
        else:
            loss = self.qup_nce_loss(batch_data)
        return loss, ia_loss, qq_training, up_loss
    def get_embeddings(self, text):
        sbert = SentenceTransformer('/home/xxxxx/TeCQR/prelm/all-MiniLM-L6-v2')
        return sbert.encode(text)
    def tag_enhanced_two_stage_offline_training(self, batch_data):
        """
        q+u->i; i->a; i+a->v or -v
        """
        user_idxs, product_idxs, query_word_idxs, _, \
        wrapped_neg_idxs, aspect_value_entries = batch_data
        av_u_neg_pidxs, p_neg_aspect_idxs, pa_neg_value_idxs, u_neg_aspect_idxs \
            = map(self.convert2cuda_lt, wrapped_neg_idxs[3:])
        aspect_idxs = [x[0] for x in aspect_value_entries if x]  
        value_idxs = [x[2] for x in aspect_value_entries if x] 
        value_symbols = [x[3] for x in aspect_value_entries if x]  
        corres_product_idxs = [product_idxs[i] for i in range(len(product_idxs)) if aspect_value_entries[i]]
        corres_user_idxs = [user_idxs[i] for i in range(len(user_idxs)) if aspect_value_entries[i]]
        valid_idxs = [i for i in range(len(product_idxs)) if aspect_value_entries[i]]
        user_idxs, product_idxs, corres_product_idxs, corres_user_idxs,  valid_idxs, query_word_idxs\
            = map(self.convert2cuda_lt, \
                  [user_idxs, product_idxs, corres_product_idxs, corres_user_idxs, valid_idxs, query_word_idxs])
        query_weight = self.query_weight_emb(query_word_idxs)
        query_vecs = self.product_emb(query_word_idxs)
        query_expanded = query_weight.expand(-1, query_vecs.size(1))
        product_vecs = self.product_emb(corres_product_idxs)
        product_weight = self.query_weight_emb(corres_product_idxs)
        product_expanded = product_weight.expand(-1, query_vecs.size(1))
        if self.gpu:
            product_vecs = product_vecs.cuda()
        tq_training= torch.tensor(0.).cuda() if self.av_word_emb.weight.is_cuda else torch.tensor(0.)
        up_loss = torch.tensor(0.).cuda() if self.av_word_emb.weight.is_cuda else torch.tensor(0.)
        if len(aspect_idxs) > 0:
            aspect_idxs, value_idxs = map(self.convert2cuda_lt, [aspect_idxs, value_idxs])
            value_symbols = self.convert2cuda_ft(value_symbols)
            aspect_vecs = self.get_embedding_from_words_asp(
                self.av_word_emb, self.aspect_keys[aspect_idxs], self.av_padding_idx)
            if self.gpu:
                aspect_vecs = aspect_vecs.cuda()
            asp_weight = self.asp_weight_emb(aspect_idxs)
            asp_expanded = asp_weight.expand(-1, aspect_vecs.size(1))
            allasp_weight = 1 - query_weight
            allasp_expanded = allasp_weight.expand(-1, aspect_vecs.size(1))
            if "av" in self.likelihood_way:
                if not self.loss_ablation == "ia":
                    tq_training= self.nce_loss(product_vecs * product_expanded, self.aspect_dists,
                                            aspect_idxs, self.av_word_emb, self.aspect_bias,
                                            idx2target=self.aspect_keys, neg_sample_idxs=p_neg_aspect_idxs)
                    qq_training = self.nce_loss(query_vecs * query_expanded, self.aspect_dists,
                                            aspect_idxs, self.av_word_emb, self.aspect_bias,
                                            idx2target=self.aspect_keys, neg_sample_idxs=u_neg_aspect_idxs)
                query_vecs = query_vecs * query_expanded + aspect_vecs * asp_expanded
                if self.gpu:
                    query_vecs = query_vecs.cuda()
        if "ori" in self.likelihood_way or "item" in self.likelihood_way:
            up_loss = self.nce_loss(query_vecs, self.product_dists,
                                    product_idxs, self.product_emb, self.product_bias,
                                    neg_sample_idxs=av_u_neg_pidxs)
        self.group_tq_training+= tq_training.item()
        self.group_iav_loss += qq_training.item()
        self.group_up_loss1 += up_loss.item()
        lam = torch.sigmoid(self.als_param).to(tq_training.device)
        loss = lam * tq_training + (1 - lam) * qq_training + up_loss
        return loss, tq_training, qq_training, up_loss
    def nce_loss(self, source_vectors, weight_dists, target_idxs,
                 target_emb, target_bias, neg_sample_idxs=None,
                 idx2target=None, sim_func='product'):
        start_time = time.time()
        context_size = 1
        target_size = target_bias.size()[0]  
        batch_size = source_vectors.size()[0]
        if neg_sample_idxs is None:
            if weight_dists is not None:
                neg_sample_idxs = torch.multinomial(weight_dists, batch_size * context_size * self.n_negs,
                                                    replacement=True)
            else:
                if source_vectors.is_cuda:
                    neg_sample_idxs = FloatTensor(batch_size * context_size * self.n_negs, device='gpu:0').uniform_(0,
                                                                                                                    target_size - 1).long()
                else:
                    neg_sample_idxs = FloatTensor(batch_size * context_size * self.n_negs, device='cpu').uniform_(0,
                                                                                                                  target_size - 1).long()
        else:
            neg_sample_idxs = neg_sample_idxs.view(-1)
        self.nce_neg_gen_time += time.time() - start_time
        start_time = time.time()
        neg_sample_idxs = neg_sample_idxs.to("cpu")
        target_idxs = target_idxs.to("cpu")
        neg_bias = target_bias[neg_sample_idxs].view(batch_size, -1)
        true_bias = target_bias[target_idxs].view(batch_size, -1)
        if self.gpu:
            neg_sample_idxs = neg_sample_idxs.cuda()
            target_idxs = target_idxs.cuda()
        if idx2target is not None:
            target_vectors = self.get_embedding_from_words_asp(
                target_emb, idx2target[target_idxs], self.av_padding_idx)
            target_vectors = target_vectors.view(batch_size, 1, -1)
            neg_vectors = self.get_embedding_from_words_asp(
                target_emb, idx2target[neg_sample_idxs], self.av_padding_idx)
            nvectors = neg_vectors.view(batch_size, context_size * self.n_negs, -1)
        else:
            target_vectors = target_emb(target_idxs).unsqueeze(1)
            neg_sample_idxs = neg_sample_idxs.view(batch_size, -1)
            nvectors = target_emb(neg_sample_idxs)
        self.nce_get_vec_time += time.time() - start_time
        start_time = time.time()
        self.nce_unsqueeze_time += time.time() - start_time
        start_time = time.time()
        if self.gpu:
            target_vectors = target_vectors.cuda()
            source_vectors = source_vectors.cuda()
        nvectorss = nvectors.neg()
        if self.gpu:
            nvectorss = nvectorss.cuda()
        oloss = torch.bmm(target_vectors, source_vectors.unsqueeze(2)).squeeze(-1)
        nloss = torch.bmm(nvectorss, source_vectors.unsqueeze(2)).squeeze(-1)
        if self.gpu:
            oloss = oloss.cuda()
            nloss = nloss.cuda()
        self.nce_bmm_time += time.time() - start_time
        start_time = time.time()
        if sim_func == 'bias_product':
            oloss = oloss + true_bias
            nloss = nloss + neg_bias
        oloss = oloss.sigmoid().log() 
        nloss = nloss.sigmoid().log().sum(1)  
        loss = -(oloss + nloss).mean()
        self.nce_compute_loss_time += time.time() - start_time
        return loss
