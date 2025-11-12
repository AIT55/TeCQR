import numpy as np
import json
import random
import gzip
import math
import collections as colls
import pickle
import copy
import time
import torch
from collections import Counter
from sentence_transformers import SentenceTransformer
import os
import pickle
import json
import torch
import torch.nn as nn
from recommendersystem.recsys import recsys
class noise_tolerance(nn.Module):
    def __init__(self):
        super(noise_tolerance, self).__init__()
        self.fc1 = nn.Linear(384 * 2, 128)  
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, query_embed, tag_embed):
        query_embed = query_embed.unsqueeze(0) if query_embed.dim() == 1 else query_embed
        tag_embed = tag_embed.unsqueeze(0) if tag_embed.dim() == 1 else tag_embed
        x = torch.cat((query_embed, tag_embed), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()  
class upsearch_data:
    def __init__(self, data_path, input_train_dir,
            set_name, batch_size, model_struct,
            threshold='strict', info_level = 3, Qselection='GBS', iter_count=10,
            is_av_embed_shared = False,
            is_feedback_same_user = False,
            keep_feedback_type = 'pos_neg',
            feedback_user_model = 'first',
            n_negs = 3,
            max_av_count = 1,
            neg_per_pos = 3, unique_att_num=2278, remove_pos_samples = False):
        self.query_max_length=200
        self.unique_att_num=unique_att_num
        self.questionsIndex=dict()
        self.sel_av_pairs=dict()
        self.sel_av_pairs_allin = dict()
        self.rr=dict()
        self.Qselection=Qselection
        self.iter_count = iter_count
        self.X_X=dict()
        self.Y_Y=dict()
        self.del_vec=dict()
        self.info_level = info_level
        self.is_feedback_same_user = is_feedback_same_user
        self.is_av_embed_shared = is_av_embed_shared
        self.keep_feedback_type = keep_feedback_type
        self.feedback_user_model = feedback_user_model
        self.batch_size = batch_size
        self.neg_per_pos = neg_per_pos
        self.n_negs = n_negs
        self.model_struct = model_struct
        self.max_av_count = max_av_count 
        self.product_ids = []
        aspects_words=dict()
        self.product_query_idx = []
        self.product_query_idx_dict = {}
        self.product_new_ids = {}
        self.user_new_ids = {}
        data_path = './data/lastfm/Graph_generate_data'
        with open('%s/question_new_num.txt' % data_path, 'r',encoding='UTF-8') as file:
            self.old_new = {}
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    self.old_new[int(parts[0])] = int(parts[1])
        with open('%s/all_pro.json' % data_path, 'r') as f, open('%s/pos_query_json_new.json' % data_path, 'r') as f1:
            f_data = json.load(f)
            f1_data = json.load(f1)
            for pid in f_data.keys():
                aspects_words[pid] = f_data[pid]["feature_index"]
                self.product_ids.append("p" + pid)
                self.product_new_ids[pid] = pid
                try:
                    values = f1_data[pid]
                except KeyError:
                    values = "0"
                for value in values:
                    self.product_query_idx.append(value)
                    if pid in self.product_query_idx_dict:
                        self.product_query_idx_dict[pid].append(value)
                    else:
                        self.product_query_idx_dict[pid] = [value]
        print("{} products loaded.".format(self.product_size))
        self.user_ids = []
        with open('%s/query_pos_json_new.json' % data_path, 'r') as f:
            f_data = json.load(f)
            for uid in f_data.keys():
                self.user_ids.append("u" + uid)
                self.user_new_ids[uid] = uid
        self.user_size = len(self.user_ids)
        print("{} users loaded.".format(self.user_size))
        self.remove_pos_samples=remove_pos_samples
        self.words = []
        with open("%s/vocab.txt" % data_path, 'rt') as fin:
            for line in fin:
                line = line.strip()
                self.words.append(line)
        self.not_idx = 2278
        self.words.append('no')
        self.vocab_size = len(self.words)
        self.padding_idx = 4556
        self.query_words = []
        count_query_temp=[]
        quiry_words_dict_nopaading={}
        with open("%s/query_words(num)_new.txt" % data_path, 'rt', encoding='UTF-8') as file:
            qury_words_dict = {}
            for line in file:
                parts = line.strip().split('\t')
                key = parts[0]
                values = list(map(int, parts[1].split()))
                count_query_temp.append(len(values))
                if len(values) > self.query_max_length:
                    values=values[:self.query_max_length]
                qury_words_dict[key] = values
                quiry_words_dict_nopaading[key] = values
        self.qury_words_dict = qury_words_dict
        self.word_count = 0
        self.vocab_distribute = np.zeros(self.vocab_size)
        self.review_info = []
        self.review_text = []
        self.item_test_ids = {}
        with open('%s/one_dict_test_new.json' % data_path, 'r') as file:
            data = json.load(file)
        for key, value in data.items():
            self.item_test_ids[key] = value["can"]
        if set_name== 'test':
            with open('%s/test.json' % data_path, 'r') as f:
                f_data = json.load(f)
                for uid in f_data.keys():
                    for pid in f_data[uid]["pos"]:
                        self.review_info.append((int(uid), pid))
                        self.review_text.append(quiry_words_dict_nopaading[str(pid)])
                        self.word_count += len(self.review_text[-1])
        else:
            with open('%s/train.json' % data_path, 'r') as f:
                f_data = json.load(f)
                for uid in f_data.keys():
                    for pid in f_data[uid]:
                        self.review_info.append((int(uid), pid))
                        self.review_text.append(quiry_words_dict_nopaading[str(pid)])
                        self.word_count += len(self.review_text[-1])
        self.review_size = len(self.review_info)
        print("{} user-product interactions loaded.".format(self.review_size))
        self.vocab_distribute = self.vocab_distribute.tolist()
        self.sub_sampling_rate = None
        self.product_distribute = np.ones(self.product_size).tolist()
        self.product_dists = self.neg_distributes(self.product_distribute)
        print("Data statistic: vocab %d, user %d, product %d\n" % (self.vocab_size,
                    self.user_size, self.product_size))
        self.value_dists = []
        self.aspect_dists = []
        self.aspect_value_count_dic = dict()
        self.av_word2id = dict()
        self.av_id2word = list()
        self.allaspects=list()
        self.aword_idx2_aid = dict()
        self.aspect_keys = []
        self.value_keys = []
        self.read_overall_aspect_values(data_path)
        if set_name == 'test' or set_name == 'valid':
            self.test_product_av_dic = dict()
            self.read_av_of_reviews_newtest(self.test_product_av_dic, input_train_dir, data_path, 'test')
        self.product_av_dic = dict()
        self.read_av_of_reviews(self.product_av_dic, input_train_dir, data_path, 'train')
    def read_overall_aspect_values(self, data_path):
        self.av_word2id = dict()
        self.allaspects=list()
        with open('%s/question_tag.json' % data_path) as f:
            f_data = json.load(f)
            for iid in f_data.keys():
                aspect_list = f_data[iid]["feature_index"]
                for aid in aspect_list:
                    if aid != 'None':
                        aid = str(aid)
                        if aid not in self.allaspects:
                            self.allaspects.append(aid)
                        if aid not in self.av_word2id:
                            self.av_word2id[aid] = int(aid)
        for val_asp in self.allaspects:
            word = val_asp + ' ' + str(self.not_idx)
            word = word.strip()
            if word not in self.av_word2id:
                if val_asp != 'None':
                    self.av_word2id[word] = int(val_asp)+self.unique_att_num
        values = [self.not_idx]
        for vid in values:
            if vid not in self.av_word2id:
                self.av_word2id[vid] = len(self.av_word2id)
        self.av_id2word = list()
        av_word2id_temp={v:k for k,v in self.av_word2id.items()}
        for i in range(len(self.av_word2id)):
            self.av_id2word.append(av_word2id_temp[i])
        if self.is_av_embed_shared:
            self.av_padding_idx = self.padding_idx
        else:
            self.av_vocab_size = len(self.av_word2id)
            self.av_padding_idx = self.av_vocab_size
    def read_av_of_reviews(self, product_av_dic, input_train_dir, data_path, set_name):
        self.value_count_dic = colls.defaultdict(float)
        self.aspect_value_count_dic = colls.defaultdict(dict)
        self.train_asp= set()
        self.max_a_length = 0
        self.max_v_length = 0
        self.asp_count_dict={}
        tmp_user_prefer_attribute = {}
        tmp_product_attribute = {}
        with open('%s/item_dict(num)_new.json' % data_path, 'r') as f:
            f_data = json.load(f)
            for pid in f_data.keys():
                tmp_product_attribute[pid] = f_data[pid]["feature_index"]
        max_aid = 0
        max_vid = 0
        with open('%s/%s.json' % (data_path, set_name), 'r') as f:
            f_data = json.load(f)
            for uid in f_data.keys():
                uid = int(uid)
                max_aid = 0
                for pid in f_data[str(uid)]:
                    yes_aspect_list = []
                    av = dict()
                    for aid in tmp_product_attribute[str(pid)]:
                        if aid != 'None':
                            yes_aspect_list.append(str(aid))
                            if int(aid) > max_aid:
                                max_aid = int(aid)
                    no_aspect_list = [i for i in self.allaspects if i not in yes_aspect_list]
                    aspect_value_list = []
                    for i in yes_aspect_list:
                        aspect_value_list.append(i)
                    for j in no_aspect_list:
                        aspect_value_list.append(j + "|"+str(self.not_idx))
                    for x in aspect_value_list:
                        a = x.split('|')[0]
                        if tuple([x.split('|')[0]]) not in self.asp_count_dict:
                            self.asp_count_dict[tuple([x.split('|')[0]])] = 0
                        self.asp_count_dict[tuple([x.split('|')[0]])] += 1
                        self.train_asp.add(a)
                        if len(x.split('|'))>1:
                            v = x.split('|')[1]
                            if v == str(self.not_idx):
                                a = a + ' ' + v
                                a = a.strip()
                                a_a = [self.av_word2id[a]]
                                v_v = [self.av_word2id[a]]
                        else:
                            a_a = [self.av_word2id[a.strip()]]
                            v_v = [self.av_word2id[a.strip()]]
                        a = tuple(a_a)
                        v = tuple(v_v)
                        self.max_v_length = max(self.max_v_length, len(v))
                        av[a] = v
                        v = v[0]  
                        if max_vid < int(v):
                            max_vid = int(v)
                        self.max_a_length = max(self.max_a_length, len(a))
                        self.value_count_dic[v] += 1
                        if v not in self.aspect_value_count_dic[a]:
                            self.aspect_value_count_dic[a][v] = 0.
                        self.aspect_value_count_dic[a][v] += 1.
                    if pid not in product_av_dic:
                        product_av_dic[pid] = dict()
                    product_av_dic[pid][uid] = av
        print("CHECK Maximum value ID: {}".format(max_vid))
    def read_av_of_reviews_newtest(self, product_av_dic, input_train_dir, data_path, set_name):
        self.value_count_dic = colls.defaultdict(float)
        self.aspect_value_count_dic = colls.defaultdict(dict)
        self.train_asp=set()
        self.max_a_length = 0
        self.max_v_length = 0
        self.asp_count_dict={}
        tmp_user_prefer_attribute = {}
        tmp_product_attribute = {}
        with open('%s/item_dict(num)_new.json' % data_path, 'r') as f:
            f_data = json.load(f)
            for pid in f_data.keys():
                tmp_product_attribute[pid] = f_data[pid]["feature_index"]
        max_aid = 0
        max_vid = 0
        with open('%s/one_dict_%s_new.json' % (data_path, set_name), 'r') as f:
            f_data = json.load(f)
            for uid in f_data.keys():
                uid = int(uid)
                max_aid = 0
                for pid in f_data[str(uid)]["pos"]:
                    yes_aspect_list = []
                    av = dict()
                    for aid in tmp_product_attribute[str(pid)]:
                        if aid != 'None':
                            yes_aspect_list.append(str(aid))
                            if int(aid) > max_aid:
                                max_aid = int(aid)
                    no_aspect_list = [i for i in self.allaspects if i not in yes_aspect_list]
                    aspect_value_list = []
                    for i in yes_aspect_list:
                        aspect_value_list.append(i)
                    for j in no_aspect_list:
                        aspect_value_list.append(j + "|"+str(self.not_idx))
                    for x in aspect_value_list:
                        a = x.split('|')[0]
                        if tuple([x.split('|')[0]]) not in self.asp_count_dict:
                            self.asp_count_dict[tuple([x.split('|')[0]])] = 0
                        self.asp_count_dict[tuple([x.split('|')[0]])] += 1
                        self.train_asp.add(a)
                        if len(x.split('|'))>1:
                            v = x.split('|')[1]
                            if v == str(self.not_idx):
                                a = a + ' ' + v
                                a = a.strip()
                                a_a = [self.av_word2id[a]]
                                v_v = [self.av_word2id[a]]
                        else:
                            a_a = [self.av_word2id[a.strip()]]
                            v_v = [self.av_word2id[a.strip()]]
                        a = tuple(a_a)
                        v = tuple(v_v)
                        self.max_v_length = max(self.max_v_length, len(v))
                        av[a] = v
                        v = v[0]
                        if max_vid < int(v):
                            max_vid = int(v)
                        self.max_a_length = max(self.max_a_length, len(a))
                        self.value_count_dic[v] += 1
                        if v not in self.aspect_value_count_dic[a]:
                            self.aspect_value_count_dic[a][v] = 0.
                        self.aspect_value_count_dic[a][v] += 1.
                    if pid not in product_av_dic:
                        product_av_dic[pid] = dict()
                    product_av_dic[pid][uid] = av
        print("CHECK Maximum value ID: {}".format(max_vid))
    def read_av_of_reviews_test(self, product_av_dic, input_train_dir, set_name):
        self.value_count_dic = colls.defaultdict(float)
        self.aspect_value_count_dic = colls.defaultdict(dict)
        self.max_a_length = 0
        self.max_v_length = 0
        with gzip.open("%s/av.%s.txt.gz" % (input_train_dir, set_name), 'rt') as fin:
            for line in fin:
                up, avs = line.strip().split(",")
                u, p = [int(x) for x in up.split("@")]
                av = dict()
                yes_aspect=[i.split('|')[0] for i in avs.split(":")]
                no_aspect_value_list=[i+"|"+str(self.not_idx) for i in self.allaspects if i not in yes_aspect]
                no_aspect_value = no_aspect_value_list
                aspect_value_list=avs.split(":")+no_aspect_value
                for x in aspect_value_list:
                    a = x.split('|')[0]
                    v = x.split('|')[1]
                    if v == str(self.not_idx):
                        a=a+' '+v
                        a = a.strip()
                        if not self.is_av_embed_shared:
                            a_a = [self.av_word2id[a]]
                            v_v = [self.av_word2id[a]]
                    else:
                        if not self.is_av_embed_shared:
                            a_a = [self.av_word2id[a.strip()], self.av_word2id[v.strip()]]
                            v_v = [self.av_word2id[v.strip()]]
                    a = tuple(a_a)
                    v = tuple(v_v)
                    self.max_v_length = max(self.max_v_length, len(v))
                    if len(v) > 2:
                        v = v[:2]
                    if len(v) > 1:
                        if not v[0] == self.av_word2id[self.not_idx]:
                            continue
                        else:
                            print("v>q1")
                            av[a] = v
                            v = v[1]
                    else:
                        av[a] = v
                        v = v[0]
                    self.max_a_length = max(self.max_a_length, len(a))
                    self.value_count_dic[v] += 1
                    if v not in self.aspect_value_count_dic[a]:
                        self.aspect_value_count_dic[a][v] = 0.
                    self.aspect_value_count_dic[a][v] += 1.
                if p not in product_av_dic:
                    product_av_dic[p] = dict()
                product_av_dic[p][u] = av
    def construct_aspect_value_keys_from_train(self):
        self.aspect_keys = list(self.aspect_value_count_dic.keys())
        self.aspect_distr = [sum([self.aspect_value_count_dic[x][y] \
            for y in self.aspect_value_count_dic[x]]) \
            for x in self.aspect_keys]
        av_pair_count = sum([len(self.aspect_value_count_dic[x]) \
            for x in self.aspect_keys])
        self.aword_idx2_aid = dict()
        for a_id in range(len(self.aspect_keys)):
            self.aword_idx2_aid[self.aspect_keys[a_id]] = self.aspect_keys[a_id][0]
        self.value_keys = list(set(list(self.value_count_dic.keys())))
        self.value_distr = [self.value_count_dic[x] for x in self.value_keys]
        self.aspect_keys_len = np.asarray([len(x) for x in self.aspect_keys])
        self.vword_idx2_vid = dict()
        for v_id in range(len(self.value_keys)):
            self.vword_idx2_vid[self.value_keys[v_id]] = self.value_keys[v_id]
        for i in range(len(self.aspect_keys)):
            self.aspect_keys[i] = list(self.aspect_keys[i]) \
                    + [self.av_padding_idx] * (self.max_a_length - len(self.aspect_keys[i]))
        self.aspect_keys.append([self.av_padding_idx] * self.max_a_length)
        self.aspect_keys = np.asarray(self.aspect_keys)
        self.aspect_distr.append(0.)
        self.value_keys.append(self.av_padding_idx)
        self.value_distr.append(0.)
        self.value_keys = list(set(self.value_keys))
        self.value_keys = np.asarray(self.value_keys)
    def sub_sampling(self, subsample_threshold):
        if subsample_threshold == 0.0:
            self.sample_count=0
            return
        self.sub_sampling_rate = [1.0 for _ in range(self.vocab_size)]
        threshold = sum(self.vocab_distribute) * subsample_threshold
        count_sub_sample = 0
        for i in range(len(self.sub_sampling_rate)):
            try:
                if self.vocab_distribute[i] == 0:
                    self.sub_sampling_rate[i] = 0
                    continue
                try:
                    self.sub_sampling_rate[i] = min(
                        (np.sqrt(float(self.vocab_distribute[i]) / threshold) + 1) * threshold / float(
                            self.vocab_distribute[i]),1.0)
                except IndexError:
                    pass
            except IndexError:
                pass
            count_sub_sample += 1
        self.sample_count = sum([self.sub_sampling_rate[i] * self.vocab_distribute[i] for i in range(len(self.sub_sampling_rate))])
    def get_av_count_dic_for_u_plist(self, product_list, uid=None):
        av_dic = colls.defaultdict(dict)
        for p in product_list:
            if p not in self.product_av_dic:
                continue
            if uid is not None:
                if uid not in self.product_av_dic[p]:
                    continue
                for a in self.product_av_dic[p][uid]:
                    if a not in self.aword_idx2_aid:
                        continue
                    v = self.product_av_dic[p][uid][a]
                    v_nosign = v[-1]
                    if v_nosign not in self.vword_idx2_vid:
                        continue
                    if v not in av_dic[self.aword_idx2_aid[a]]:
                        av_dic[self.aword_idx2_aid[a]][v] = 0
                    av_dic[self.aword_idx2_aid[a]][v] += 1
            else:
                for each_u in self.product_av_dic[p]:
                    for a in self.product_av_dic[p][each_u]:
                        if a not in self.aword_idx2_aid:
                            continue
                        v = self.product_av_dic[p][each_u][a]
                        v_nosign = v[-1]
                        if v_nosign not in self.vword_idx2_vid:
                            continue
                        if v not in av_dic[self.aword_idx2_aid[a]]:
                            av_dic[self.aword_idx2_aid[a]][v] = 0
                        av_dic[self.aword_idx2_aid[a]][v] += 1
        return av_dic
    def item_pair_to_AVs(self, posP_list, negP_list, uid=None, info_level = 1, is_test=False):
        pos_av_dic = self.get_av_count_dic_for_u_plist(posP_list, uid)
        if not pos_av_dic:
            return []
        av_pairs = []
        av_pairs_add=[]
        if info_level > 2:
            for aspect in pos_av_dic:
                for val in pos_av_dic[aspect]:
                    if val[0] != self.av_word2id[self.not_idx]:
                        if len(val) > 1 and self.av_word2id[self.not_idx]:
                            av_pairs.append([aspect, pos_av_dic[aspect][val], self.vword_idx2_vid[val[1]], -1])
                        else:
                            av_pairs.append([aspect, pos_av_dic[aspect][val], self.vword_idx2_vid[val[0]], 1])
                    else:
                        if len(val) > 1 and self.av_word2id[self.not_idx]:
                            av_pairs_add.append([aspect, pos_av_dic[aspect][val], self.vword_idx2_vid[val[1]], -1])
                        else:
                            av_pairs_add.append([aspect, pos_av_dic[aspect][val], self.vword_idx2_vid[val[0]], 1])
        av_pairs=av_pairs+av_pairs_add
        return av_pairs
    def initialize_epoch(self):
        random.shuffle(self.train_seq)
        self.review_size = len(self.train_seq)
        self.estimated_entry_size = int(self.sample_count)  + self.word_count*4
        self.neg_sample_products = np.random.randint(0, self.product_size, size = (self.review_size, self.neg_per_pos))
        self.u_neg_word_sample = np.random.choice(len(self.word_dists),
                size = (self.estimated_entry_size, self.n_negs), replace=True, p=self.word_dists)
        self.p_neg_word_sample = np.random.choice(len(self.word_dists),
                size = (self.estimated_entry_size, self.n_negs), replace=True, p=self.word_dists)
        self.u_neg_product_sample = np.random.choice(len(self.product_dists),
                size = (self.estimated_entry_size, self.n_negs), replace=True, p=self.product_dists)
        self.u_neg_product_av_sample = np.random.choice(len(self.product_dists),
                size = (self.estimated_entry_size, self.n_negs), replace=True, p=self.product_dists)
        self.p_neg_aspect_sample = np.random.choice(len(self.aspect_dists),
                size = (self.estimated_entry_size, self.n_negs), replace=True, p=self.aspect_dists)
        self.u_neg_aspect_sample = np.random.choice(len(self.aspect_dists),
                                                    size=(self.estimated_entry_size, self.n_negs), replace=True, p=self.aspect_dists)
        self.pa_neg_value_sample = np.random.choice(len(self.value_dists),
                size = (self.estimated_entry_size, self.n_negs), replace=True, p=self.value_dists)
        self.cur_review_i = 0
        self.cur_word_i = 0
        self.cur_epoch_word_i = 0
        self.cur_entry_i = 0
    def get_av_train_batch(self):
        user_idxs, product_idxs, word_idxs, aspect_value_entries = [],[],[],[]
        corres_product_idxs = []
        u_neg_words_idxs, p_neg_words_idxs, u_neg_p_idxs = [],[],[]
        av_u_neg_pidxs, p_neg_aspect_idxs, pa_neg_value_idxs, u_neg_aspect_idxs = [],[],[],[]
        query_word_idxs, corres_query_word_idxs = [], []
        query_idxs = []
        query_word_idxs_dict = {}
        review_idx = self.train_seq[self.cur_review_i]
        user_idx = self.review_info[review_idx][0]
        product_idx = self.review_info[review_idx][1]
        query_idx = str(random.choice(self.product_query_idx_dict[product_idx]))
        text_list = self.review_text[review_idx]
        text_length = len(text_list)
        neg_product_idxs = self.neg_sample_products[self.cur_review_i]
        if self.is_feedback_same_user:
            av_pairs = self.item_pair_to_AVs([product_idx], neg_product_idxs, uid=user_idx, info_level=self.info_level)
        else:
            av_pairs = self.item_pair_to_AVs([product_idx], neg_product_idxs, uid=None, info_level=self.info_level)
        time_flag = time.time()
        while len(word_idxs) < self.batch_size:
            if self.sub_sampling_rate == None or random.random() < self.sub_sampling_rate[text_list[self.cur_word_i % text_length]]:
                user_idxs.append(user_idx)
                u_neg_words_idxs.append(self.u_neg_word_sample[self.cur_entry_i % self.estimated_entry_size])
                product_idxs.append(int(product_idx))
                p_neg_words_idxs.append(self.p_neg_word_sample[self.cur_entry_i % self.estimated_entry_size])
                u_neg_p_idxs.append(self.u_neg_product_sample[self.cur_entry_i % self.estimated_entry_size])
                query_idxs.append(int(query_idx))
                word_idxs.append(text_list[self.cur_word_i % text_length])
                av_u_neg_pidxs.append(self.u_neg_product_av_sample[self.cur_entry_i % self.estimated_entry_size])
                av_pairs_length=len(av_pairs)
                if av_pairs_length > 0:
                    aspect_value = av_pairs[self.cur_word_i % av_pairs_length]
                    aspect_value_entries.append(aspect_value)
                    p_neg_aspect_idxs.append(self.p_neg_aspect_sample[self.cur_entry_i % self.estimated_entry_size])
                    u_neg_aspect_idxs.append(self.u_neg_aspect_sample[self.cur_entry_i % self.estimated_entry_size])
                    pa_neg_value_idxs.append(self.pa_neg_value_sample[self.cur_entry_i % self.estimated_entry_size])
                else:
                    aspect_value_entries.append(None)
                self.cur_entry_i += 1
            self.cur_word_i += 1
            self.cur_epoch_word_i += 1
            self.finished_word_num += 1
            if self.cur_word_i == text_length:
                self.cur_review_i += 1
                if self.cur_review_i == self.review_size:
                    break
                self.cur_word_i = 0
                review_idx = self.train_seq[self.cur_review_i]
                user_idx = self.review_info[review_idx][0]
                product_idx = self.review_info[review_idx][1]
                query_idx = str(random.choice(self.product_query_idx_dict[product_idx]))
                text_list = self.review_text[review_idx]
                text_length = len(text_list)
                neg_product_idxs = self.neg_sample_products[self.cur_review_i]
                if self.is_feedback_same_user:
                    av_pairs = self.item_pair_to_AVs([product_idx], neg_product_idxs, uid=user_idx, info_level=self.info_level)
                else:
                    av_pairs = self.item_pair_to_AVs([product_idx], neg_product_idxs, uid=None, info_level=self.info_level)
        has_next = False if self.cur_review_i == self.review_size else True
        wrapped_neg_idxs = [u_neg_words_idxs, p_neg_words_idxs,
                u_neg_p_idxs, av_u_neg_pidxs, p_neg_aspect_idxs, pa_neg_value_idxs, u_neg_aspect_idxs]
        return user_idxs, product_idxs, query_idxs, word_idxs, wrapped_neg_idxs,\
                aspect_value_entries, has_next
    def select_av_pairs_GBS(self, cur_iter_i, u_ranklist_map, max_av_count, entity_Doc_Matrix, docList, enList, product_idx, p_av_dic, user_idx, query_idx):
        query_product_tensor_loaded = torch.load('/home/xxxxxxxx/TeCQR/vscodeuse/vscodeusequery_tensor.pt')
        tag_tensor_loaded = torch.load('/home/xxxxxxxx/TeCQR/vscodeuse/vscodeusetag_tensor.pt')
        query_new_dict = {}
        with open('/home/xxxxxxxx/TeCQR/data/lastfm/Graph_generate_data/question_new_num.txt', 'r') as f:
            for line in f:
                old_id, new_id = line.strip().split('\t')
                query_new_dict[int(old_id)] = int(new_id)
        query_text = {}
        with open('/home/xxxxxxxx/TeCQR/data/lastfm/Graph_generate_data/text_tokenized.txt', 'r') as f:
            for line in f:
                old_id, text, _ = line.strip().split('\t')
                query_text[int(old_id)] = text
        tag_dict = {}
        with open('/home/xxxxxxxx/TeCQR/data/lastfm/Graph_generate_data/tag_num.txt', 'r') as f:
            for line in f:
                tag, tag_id = line.strip().split('\t')
                tag_dict[int(tag_id)] = tag
        model = noise_tolerance()
        load_model(model, "/home/xxxxxxxx/TeCQR/noisy_user_model/noisy_model_test_new.pth")
        random_bias=0.0
        u_q_p=(user_idx, query_idx, product_idx)
        if u_q_p not in self.sel_av_pairs:
            self.sel_av_pairs[u_q_p]=[]
        if u_q_p not in self.questionsIndex:
            self.questionsIndex[u_q_p]=[]
            self.questionsIndex[u_q_p]=self.questionsIndex[u_q_p]+self.removed_index
        if product_idx not in docList:
            print('No such product_idx')
            return self.sel_av_pairs
        PreferencePi = np.zeros(len(docList), dtype=np.float64)
        prod_pref_dict = {u_ranklist_map[j]: float(1.0 / (j + 1)) for j in range(len(u_ranklist_map))}
        for doc_idx in range(len(docList)):
            if int(docList[doc_idx]) in prod_pref_dict:
                PreferencePi[doc_idx] = prod_pref_dict[int(docList[doc_idx])]
        min = 1000000000.0
        minIndex = 0
        for z in range(len(entity_Doc_Matrix)):
            argMin = 0.0
            if z not in self.questionsIndex[u_q_p]:
                argMin = np.dot(entity_Doc_Matrix[z], PreferencePi)
                judge = abs(argMin)
                if judge < min:
                    min = judge
                    minIndex = z
        self.questionsIndex[u_q_p].append(minIndex)
        countIsZero = 0
        countIsOne = 0
        countNull = 0
        product_list=[product_idx]
        for singlerel in product_list:
            indexM = docList.index(singlerel)
            if entity_Doc_Matrix[minIndex, indexM] == -1:
                countIsZero += 1
            elif entity_Doc_Matrix[minIndex, indexM] == 1:
                countIsOne += 1
            else:
                countNull += 1
        tuple_enList = []
        if (countIsZero == countIsZero + countIsOne) and (countIsZero != 0):
            value = str(self.not_idx)
            sig_posneg = -1
            if not self.is_av_embed_shared:
                rand_value = random.random()
                probability = 0.5
                if rand_value < probability:
                    tuple_enList = [[self.av_word2id[x.strip()] for x in enList[minIndex]]]
                    query_embed = query_product_tensor_loaded[query_idx]
                    tag_embed = tag_tensor_loaded[int(enList[minIndex][0])]
                    score = model(torch.tensor(query_embed), torch.tensor(tag_embed))
                    if score <= 0.5:
                        tuple_enList = [[self.av_word2id[x.strip() + ' ' + value.strip()] for x in enList[minIndex]]]
                    else:
                        return self.select_av_pairs_GBS_suboptimal(cur_iter_i, u_ranklist_map, max_av_count, entity_Doc_Matrix, docList, enList, product_idx, p_av_dic, user_idx, query_idx)
                else:
                    tuple_enList = [[self.av_word2id[x.strip() + ' ' + value.strip()] for x in enList[minIndex]]]
                value = tuple_enList
        elif (countIsOne == countIsZero + countIsOne) and (countIsOne != 0):
            value_list = p_av_dic[product_idx][enList[minIndex]]
            sig_posneg=1
            if not self.is_av_embed_shared:
                tuple_enList_pre = [x for x in enList[minIndex]]
                value=[]
                tuple_enList=[]
                for va_l in value_list:
                    randomvalue = str(self.not_idx)
                    rand_value = random.random()
                    probability = 0.5
                    if rand_value < probability:
                        va_l_pre = [self.av_word2id[x.strip()+ ' ' + randomvalue.strip()] for x in va_l]
                        query_embed = query_product_tensor_loaded[query_idx]
                        tag_embed = tag_tensor_loaded[int(va_l[0])]
                        score = model(torch.tensor(query_embed), torch.tensor(tag_embed))
                        if score >= 0.5:
                            va_l_pre=[self.av_word2id[x.strip()] for x in va_l]
                        else:
                            return self.select_av_pairs_GBS_suboptimal(cur_iter_i, u_ranklist_map, max_av_count, entity_Doc_Matrix, docList, enList, product_idx, p_av_dic, user_idx, query_idx)
                    else:
                        va_l_pre = [self.av_word2id[x.strip()] for x in va_l]
                    tuple_enList.append(va_l_pre)
                    value.append(va_l_pre)
        else:
            pass
        countnotappear=0
        tempavpair=[]
        for val in range(len(tuple_enList)):
            tuple_enList_val = tuple(tuple_enList[val])
            aword_idx2_aid = self.aword_idx2_aid
            if tuple_enList_val in self.aword_idx2_aid:
                aspect = self.aword_idx2_aid[tuple_enList_val] 
                tempavpair.append([aspect, 1, self.vword_idx2_vid[value[val][0]], sig_posneg])
                if len(value[val])>1:
                    print("len(val)>q1**********************************")
            else:
                countnotappear+=1
        if len(tempavpair)>0:
            random.shuffle(tempavpair)
            self.sel_av_pairs[u_q_p].append(tempavpair[0])
        else:
            self.sel_av_pairs[u_q_p].append([])
        return self.sel_av_pairs
    def select_av_pairs_GBS_suboptimal(self, cur_iter_i, u_ranklist_map, max_av_count, entity_Doc_Matrix, docList, enList, product_idx, p_av_dic, user_idx, query_idx):
        query_product_tensor_loaded = torch.load('/home/xxxxxxxx/TeCQR/vscodeuse/vscodeusequery_tensor.pt')
        tag_tensor_loaded = torch.load('/home/xxxxxxxx/TeCQR/vscodeuse/vscodeusetag_tensor.pt')
        query_new_dict = {}
        with open('/home/xxxxxxxx/TeCQR/data/lastfm/Graph_generate_data/question_new_num.txt', 'r') as f:
            for line in f:
                old_id, new_id = line.strip().split('\t')
                query_new_dict[int(old_id)] = int(new_id)
        query_text = {}
        with open('/home/xxxxxxxx/TeCQR/data/lastfm/Graph_generate_data/question_text.txt', 'r') as f:
            for line in f:
                old_id, text, _ = line.strip().split('\t')
                query_text[int(old_id)] = text
        tag_dict = {}
        with open('/home/xxxxxxxx/TeCQR/data/lastfm/Graph_generate_data/tag_num.txt', 'r') as f:
            for line in f:
                tag, tag_id = line.strip().split('\t')
                tag_dict[int(tag_id)] = tag
        model = noise_tolerance()
        load_model(model, "/home/xxxxxxxx/TeCQR/noisy_user_model/noisy_model_test_new.pth")
        random_bias=0.0
        u_q_p=(user_idx, query_idx, product_idx)
        if u_q_p not in self.sel_av_pairs:
            self.sel_av_pairs[u_q_p]=[]
        if u_q_p not in self.questionsIndex:
            self.questionsIndex[u_q_p]=[]
            self.questionsIndex[u_q_p]=self.questionsIndex[u_q_p]+self.removed_index
        if product_idx not in docList:
            print('No such product_idx')
            return self.sel_av_pairs
        PreferencePi = np.zeros(len(docList), dtype=np.float64)
        prod_pref_dict = {u_ranklist_map[j]: float(1.0 / (j + 1)) for j in range(len(u_ranklist_map))}
        for doc_idx in range(len(docList)):
            if int(docList[doc_idx]) in prod_pref_dict:
                PreferencePi[doc_idx] = prod_pref_dict[int(docList[doc_idx])]
        min = 1000000000.0
        minIndex = 0
        for z in range(len(entity_Doc_Matrix)):
            argMin = 0.0
            if z not in self.questionsIndex[u_q_p]:
                argMin = np.dot(entity_Doc_Matrix[z], PreferencePi)
                judge = abs(argMin)
                if judge < min:
                    min = judge
                    minIndex = z
        self.questionsIndex[u_q_p].append(minIndex)
        countIsZero = 0
        countIsOne = 0
        countNull = 0
        product_list=[product_idx]
        for singlerel in product_list:
            indexM = docList.index(singlerel)
            if entity_Doc_Matrix[minIndex, indexM] == -1:
                countIsZero += 1
            elif entity_Doc_Matrix[minIndex, indexM] == 1:
                countIsOne += 1
            else:
                countNull += 1
        tuple_enList = []
        if (countIsZero == countIsZero + countIsOne) and (countIsZero != 0):
            value = str(self.not_idx)
            sig_posneg = -1
            if not self.is_av_embed_shared:
                rand_value = random.random()
                probability = 0.5
                if rand_value < probability:
                    query_embed = query_product_tensor_loaded[query_idx]
                    tag_embed = tag_tensor_loaded[int(enList[minIndex][0])]
                    score = model(torch.tensor(query_embed), torch.tensor(tag_embed))
                    if score <= 0.5:
                        tuple_enList = [[self.av_word2id[x.strip() + ' ' + value.strip()] for x in enList[minIndex]]]
                    else:
                        tuple_enList = [[self.av_word2id[x.strip()] for x in enList[minIndex]]]
                else:
                    tuple_enList = [[self.av_word2id[x.strip() + ' ' + value.strip()] for x in enList[minIndex]]]
                value = tuple_enList
        elif (countIsOne == countIsZero + countIsOne) and (countIsOne != 0):
            value_list = p_av_dic[product_idx][enList[minIndex]]
            sig_posneg=1
            if not self.is_av_embed_shared:
                tuple_enList_pre = [x for x in enList[minIndex]]
                value=[]
                tuple_enList=[]
                for va_l in value_list:
                    randomvalue = str(self.not_idx)
                    rand_value = random.random()
                    probability = 0.5
                    if rand_value < probability:
                        query_embed = query_product_tensor_loaded[query_idx]
                        tag_embed = tag_tensor_loaded[int(va_l[0])]
                        score = model(torch.tensor(query_embed), torch.tensor(tag_embed))
                        if score >= 0.5:
                            va_l_pre=[self.av_word2id[x.strip()] for x in va_l]
                        else:
                            va_l_pre = [self.av_word2id[x.strip()+ ' ' + randomvalue.strip()] for x in va_l]
                    else:
                        va_l_pre = [self.av_word2id[x.strip()] for x in va_l]
                    tuple_enList.append(va_l_pre)
                    value.append(va_l_pre)
        else:
            pass
        countnotappear=0
        tempavpair=[]
        for val in range(len(tuple_enList)):
            tuple_enList_val = tuple(tuple_enList[val])
            aword_idx2_aid = self.aword_idx2_aid
            if tuple_enList_val in self.aword_idx2_aid:
                aspect = self.aword_idx2_aid[tuple_enList_val]
                tempavpair.append([aspect, 1, self.vword_idx2_vid[value[val][0]], sig_posneg])
                if len(value[val])>1:
                    print("len(val)>q1**********************************")
            else:
                countnotappear+=1
        if len(tempavpair)>0:
            random.shuffle(tempavpair)
            self.sel_av_pairs[u_q_p].append(tempavpair[0])
        else:
            self.sel_av_pairs[u_q_p].append([])
        return self.sel_av_pairs
    def select_av_pairs_random(self, u_ranklist_map, max_av_count, entity_Doc_Matrix, docList, enList, product_idx, p_av_dic, user_idx, query_idx):
        u_q_p=(user_idx, query_idx, product_idx)
        if u_q_p not in self.sel_av_pairs:
            self.sel_av_pairs[u_q_p]=[]
        if u_q_p not in self.questionsIndex:
            self.questionsIndex[u_q_p]=[]
            self.questionsIndex[u_q_p]=self.questionsIndex[u_q_p]+self.removed_index
        if product_idx not in docList:
            print('No such product_idx')
            return self.sel_av_pairs
        min = 1000000000.0
        minIndex = -1
        for k in range(1000):
            if minIndex == -1:
                Randomx = random.randint(0, len(entity_Doc_Matrix) - 1)
                if Randomx not in self.questionsIndex[u_q_p]:
                    minIndex = Randomx
            else:
                break
        self.questionsIndex[u_q_p].append(minIndex)
        countIsZero = 0
        countIsOne = 0
        countNull = 0
        product_list=[product_idx]
        for singlerel in product_list:
            indexM = docList.index(singlerel)
            if entity_Doc_Matrix[minIndex, indexM] == -1:
                countIsZero += 1
            elif entity_Doc_Matrix[minIndex, indexM] == 1:
                countIsOne += 1
            else:
                countNull += 1
        tuple_enList = []
        if (countIsZero == countIsZero + countIsOne) and (countIsZero != 0):
            value = str(self.not_idx)
            sig_posneg = -1
            if not self.is_av_embed_shared:
                tuple_enList = [[self.av_word2id[x.strip()+' '+value.strip()] for x in enList[minIndex]]]
                value=tuple_enList
        elif (countIsOne == countIsZero + countIsOne) and (countIsOne != 0):
            value_list = p_av_dic[product_idx][enList[minIndex]]
            sig_posneg=1
            if not self.is_av_embed_shared:
                tuple_enList_pre = [self.av_word2id[x.strip()] for x in enList[minIndex]]
                value=[]
                tuple_enList=[]
                for va_l in value_list:
                    va_l_pre=[self.av_word2id[x.strip()] for x in va_l]
                    tuple_enList.append(tuple_enList_pre)
                    value.append(va_l_pre)
        else:
            pass
        countnotappear=0
        tempavpair=[]
        for val in range(len(tuple_enList)):
            tuple_enList_val = tuple(tuple_enList[val])
            if tuple_enList_val in self.aword_idx2_aid:
                aspect = self.aword_idx2_aid[tuple_enList_val] 
                tempavpair.append([aspect, 1, self.vword_idx2_vid[value[val][0]], sig_posneg])
                if len(value[val])>1:
                    print("len(val)>q1**********************************")
            else:
                countnotappear+=1
        if countnotappear==len(tuple_enList):
            print('aspects not appear')
        if len(tempavpair)>0:
            random.shuffle(tempavpair)
            self.sel_av_pairs[u_q_p].append(tempavpair[0])
        return self.sel_av_pairs
    def select_av_pairs_Bandit(self, max_av_count, entity_Doc_Matrix, docList, enList, product_idx, p_av_dic, user_idx, query_idx, aa, c):
        u_q_p=(user_idx, query_idx, product_idx)
        if u_q_p not in self.sel_av_pairs:
            self.sel_av_pairs[u_q_p]=[]
        if u_q_p not in self.questionsIndex:
            self.questionsIndex[u_q_p]=[]
            self.questionsIndex[u_q_p]=self.questionsIndex[u_q_p]+self.removed_index
        if u_q_p not in self.rr:
            self.rr[u_q_p]=torch.from_numpy(np.zeros(len(enList), dtype=np.float64))
        if product_idx not in docList:
            print('No such product_idx')
            return self.sel_av_pairs
        max = 0
        minIndex = 0
        for z in range(len(entity_Doc_Matrix)):
            argMax = 0.0
            if z not in self.questionsIndex[u_q_p]:
                argMax = torch.dot(aa[z], self.rr[u_q_p]) + (c / 2) * aa[z].norm(2)
                if argMax > max:
                    max = argMax
                    minIndex = z
        self.questionsIndex[u_q_p].append(minIndex)
        countIsZero = 0
        countIsOne = 0
        countNull = 0
        product_list=[product_idx]
        for singlerel in product_list:
            indexM = docList.index(singlerel)
            if entity_Doc_Matrix[minIndex, indexM] == 0:
                countIsZero += 1
            elif entity_Doc_Matrix[minIndex, indexM] == 1:
                countIsOne += 1
            else:
                countNull += 1
        tuple_enList = []
        if (countIsZero == countIsZero + countIsOne) and (countIsZero != 0):
            self.rr[u_q_p][minIndex] = -1.0
            enList_temp=[int(x) for x in enList[minIndex]]
            value = str(self.not_idx)
            sig_posneg = -1
            if not self.is_av_embed_shared:
                tuple_enList = [[self.av_word2id[x.strip()] for x in enList[minIndex]]]
                value=tuple_enList
        elif (countIsOne == countIsZero + countIsOne) and (countIsOne != 0):
            self.rr[u_q_p][minIndex] = 1.0
            value_list = p_av_dic[product_idx][enList[minIndex]]
            sig_posneg=1
            if not self.is_av_embed_shared:
                tuple_enList_pre = [self.av_word2id[x.strip()] for x in enList[minIndex]]
                value = []
                tuple_enList=[]
                for va_l in value_list:
                    va_l_pre=[self.av_word2id[x.strip()] for x in va_l]
                    tuple_enList.append(tuple_enList_pre)
                    value.append(va_l_pre)
            enList_temp=[int(x) for x in enList[minIndex]]
            vlist_temp=[int(x[0]) for x in value_list]
        else:
            pass
        countnotappear=0
        tempavpair=[]
        for val in range(len(tuple_enList)):
            tuple_enList_val = tuple(tuple_enList[val])
            if tuple_enList_val in self.aword_idx2_aid:
                aspect = self.aword_idx2_aid[tuple_enList_val]
                tempavpair.append([aspect, 1, self.vword_idx2_vid[value[val][0]], sig_posneg])
                if len(value[val])>1:
                    print("len(val)>q1**********************************")
            else:
                countnotappear += 1
        if countnotappear==len(tuple_enList):
            print('aspects not appear')
        if len(tempavpair)>0:
            random.shuffle(tempavpair)
            self.sel_av_pairs[u_q_p].append(tempavpair[0])
        return self.sel_av_pairs
    def get_av_test_batch(self, user_ranklist_map, cur_iter_i, c, entity_Doc_Matrix=None, docList=None, enList=None, p_av_dic=None, aa=None, rr=None):
        user_idxs, query_idxs, query_word_idxs = [],[],[]
        av_user_idxs, av_query_idxs, av_query_word_idxs, query_av_pairs = [],[],[],[]
        av_product_idxs = []
        cur_av_pair_count = self.max_av_count * (cur_iter_i - 1) 
        while len(user_idxs) + len(av_user_idxs) < self.batch_size \
                and self.cur_uqr_i < self.test_qu_size:
            av_pairs = []
            user_idx, product_idx, query_idx, review_idx = self.test_seq[self.cur_uqr_i]
            if cur_iter_i > 1:
                uid = None
                if self.is_feedback_same_user:
                    uid = user_idx
                tempprintdic={}
                if "GBS" == self.Qselection:
                    av_pairs_all = self.select_av_pairs_GBS(cur_iter_i, user_ranklist_map[(user_idx, query_idx)], cur_av_pair_count, entity_Doc_Matrix, docList, enList, product_idx, p_av_dic, user_idx, query_idx)
                    av_pairs=av_pairs_all[(user_idx, query_idx, product_idx)]
                if "bandit" == self.Qselection:
                    av_pairs_all = self.select_av_pairs_Bandit(cur_av_pair_count, entity_Doc_Matrix, docList, enList, product_idx, p_av_dic, user_idx, query_idx, aa, c)
                    av_pairs = av_pairs_all[(user_idx, query_idx, product_idx)]
                if "random" == self.Qselection:
                    av_pairs_all = self.select_av_pairs_random(user_ranklist_map[(user_idx, query_idx)], cur_av_pair_count, entity_Doc_Matrix, docList, enList, product_idx, p_av_dic, user_idx, query_idx)
                    av_pairs=av_pairs_all[(user_idx, query_idx, product_idx)]
            if len(av_pairs) > 0:
                av_user_idxs.append(user_idx)
                av_query_idxs.append(query_idx)
                remove_empty_av_pairs = [x for x in av_pairs[:cur_av_pair_count] if x]
                query_av_pairs.append(remove_empty_av_pairs)
                av_query_word_idxs.append(int(query_idx))
                av_product_idxs.append(int(product_idx))
            else:
                user_idxs.append(user_idx)
                query_idxs.append(query_idx)
                query_word_idxs.append(int(query_idx))
            self.cur_uqr_i += 1
        if len(query_av_pairs)> 0:
            max_avpairs_len = np.max([len(x) for x in query_av_pairs])
            for qap_idx in range(len(query_av_pairs)):
                if len(query_av_pairs[qap_idx]) < max_avpairs_len:
                    query_av_pairs[qap_idx] = query_av_pairs[qap_idx] + [[self.av_padding_idx, 0, self.av_padding_idx, -1] for _ in range(max_avpairs_len - len(query_av_pairs[qap_idx]))]
        query_av_pairs = np.asarray(query_av_pairs)
        has_next = False if self.cur_uqr_i == len(self.test_seq) else True
        return user_idxs, query_idxs, query_word_idxs, \
               av_user_idxs, av_query_idxs, av_query_word_idxs, np.asarray(query_av_pairs), av_product_idxs, has_next
    def output_ranklist(self, user_ranklist_map, user_ranklist_score_map, output_path, similarity_func):
        outfile = "%s/test.%s.ranklist" % (output_path, similarity_func)
        total_qu = 0
        with open(outfile, 'w') as rank_fout:
            for uq_pair in user_ranklist_map:
                if len(user_ranklist_map[uq_pair]) == 0:
                    continue
                total_qu += 1
                uidx, qidx = uq_pair
                user_id = self.user_new_ids[str(uidx)]
                for i in range(len(user_ranklist_map[uq_pair])):
                    product_id = user_ranklist_map[uq_pair][i]
                    product_id = self.product_new_ids[str(product_id)]
                    if product_id in self.item_test_ids[user_id]:
                        line = "%s_%s Q0 %s %d %f RetrievalEmbedding\n" % (user_id, qidx, product_id, i+1, user_ranklist_score_map[uq_pair][i])
                        rank_fout.write(line)
        print("total qu:", total_qu)
