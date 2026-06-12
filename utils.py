import torch
import os
import numpy as np
import signal
import time
import subprocess
import torch_geometric
import torch_geometric.nn as geo_nn
from copy import deepcopy
from tqdm import tqdm
from functools import wraps
from collections import defaultdict


def load_g_graph(g_file):
    nid = list()
    nlabel = list()
    nindeg = list()
    elabel = list()
    e_u = list()
    e_v = list()
    with open(g_file) as f2:
        num_nodes = int(f2.readline().rstrip())
        v_neigh = list()
        for i in range(num_nodes):
            temp_list = list()
            v_neigh.append(temp_list)
        for i in range(num_nodes):
            node_info = f2.readline()
            node_id, node_label = node_info.rstrip().split()
            nid.append(int(node_id))
            nlabel.append(int(node_label))
        while True:
            line = f2.readline()
            # read until the end of the file.
            if not line:
                break
            temp_indeg = int(line.strip())
            nindeg.append(temp_indeg)
            if temp_indeg == 0:
                continue
            for i in range(temp_indeg):
                edge_info = f2.readline().rstrip().split()
                if len(edge_info) == 2:
                    edge_label = 1
                else:
                    edge_label = int(edge_info[-1])
                e_u.append(int(edge_info[0]))
                e_v.append(int(edge_info[1]))
                v_neigh[int(edge_info[0])].append(int(edge_info[1]))
                # v_neigh[int(edge_info[1])].append(int(edge_info[0]))
                elabel.append(edge_label)
    g_nid = deepcopy(nid)
    g_nlabel = deepcopy(nlabel)
    g_indeg = deepcopy(nindeg)
    g_edges = [deepcopy(e_u), deepcopy(e_v)]
    g_elabel = deepcopy(elabel)
    g_v_neigh = deepcopy(v_neigh)
    g_label_dict = defaultdict(list)
    for i in range(len(g_nlabel)):
        g_label_dict[g_nlabel[i]].append(i)
    graph_info = [
        g_nid,
        g_nlabel,
        g_indeg,
        g_edges,
        g_elabel,
        g_v_neigh,
        g_label_dict
    ]
    return graph_info


def load_p_data(p_file):
    nid = list()
    nlabel = list()
    nindeg = list()
    elabel = list()
    e_u = list()
    e_v = list()

    with open(p_file) as f1:
        num_nodes = int(f1.readline().rstrip())
        v_neigh = list()
        for i in range(num_nodes):
            temp_list = list()
            v_neigh.append(temp_list)
        for i in range(num_nodes):
            node_info = f1.readline()
            node_id, node_label = node_info.rstrip().split()
            # print(node_id)
            # print(node_label)
            nid.append(int(node_id))
            nlabel.append(int(node_label))
        while True:
            line = f1.readline()
            # read until the end of the file.
            if not line:
                break
            temp_indeg = int(line.strip())
            nindeg.append(temp_indeg)
            if temp_indeg == 0:
                continue
            for i in range(temp_indeg):
                edge_info = f1.readline().rstrip().split()
                if len(edge_info) == 2:
                    edge_label = 1
                else:
                    edge_label = int(edge_info[-1])
                e_u.append(int(edge_info[0]))
                e_v.append(int(edge_info[1]))
                v_neigh[int(edge_info[0])].append(int(edge_info[1]))
                # v_neigh[int(edge_info[1])].append(int(edge_info[0]))
                elabel.append(edge_label)
    p_nid = deepcopy(nid)
    p_nlabel = deepcopy(nlabel)
    p_indeg = deepcopy(nindeg)
    p_edges = [deepcopy(e_u), deepcopy(e_v)]
    p_elabel = deepcopy(elabel)
    p_v_neigh = [deepcopy(v_list) for v_list in v_neigh]
    p_label_dict = defaultdict(list)
    for i in range(len(p_nlabel)):
        p_label_dict[p_nlabel[i]].append(i)
    pattern_info = [
        p_nid,
        p_nlabel,
        p_indeg,
        p_edges,
        p_elabel,
        p_v_neigh,
        p_label_dict
    ]
    return pattern_info


def load_baseline(b_file):
    baseline_dict = dict()
    with open(b_file) as f:
        for line in f:
            file, true_count = line.rstrip().split()
            baseline_dict[file] = int(true_count)
    return baseline_dict


def load_iso_baseline(baseline_path, prefix):
    baseline_dict = dict()
    file_paths = os.listdir(baseline_path)
    for file_path in file_paths:
        current_path = os.path.join(baseline_path, file_path)
        file_list = os.listdir(current_path)
        for file_name in file_list:
            current_file = os.path.join(current_path, file_name)
            with open(current_file) as f:
                line_0 = f.readline().rstrip()
                baseline_dict[prefix+file_name.replace('.txt', '.graph')] = int(line_0)
    return baseline_dict


def int_to_multihot(input_int, dim):
    init = np.zeros(dim)
    binary_string = '{0:b}'.format(input_int)
    diff = dim - len(binary_string)
    for i in reversed(range(len(binary_string))):
        init[i+diff] = int(binary_string[i])
    return init


def generate_features(graph_info, vec_dim):
    # data information contains:
    # 0: id 1: label 2: degree 3: edge_info 4: edge_label 5: vertex neighbor 6: label_dict
    vertices_id = graph_info[0]
    label_info = graph_info[1]
    degree_info = graph_info[2]
    neighbor_info = graph_info[5]
    feature_vec = np.array([])
    for i in range(len(label_info)):    # num of graph vertices
        label_vec = int_to_multihot(label_info[vertices_id[i]], vec_dim)
        degree_vec = int_to_multihot(degree_info[vertices_id[i]], vec_dim)
        neigh_label_vec = np.expand_dims(np.zeros(vec_dim), axis=0)
        neigh_degree_vec = np.expand_dims(np.zeros(vec_dim), axis=0)
        for j in range(len(neighbor_info[vertices_id[i]])):
            if j == 0:
                neigh_label_vec = np.expand_dims(int_to_multihot(label_info[neighbor_info[vertices_id[i]][j]], vec_dim), axis=0)
                neigh_degree_vec = np.expand_dims(int_to_multihot(degree_info[neighbor_info[vertices_id[i]][j]], vec_dim), axis=0)
            else:
                neigh_label_vec = np.append(neigh_label_vec, np.expand_dims(int_to_multihot(label_info[neighbor_info[vertices_id[i]][j]], vec_dim), axis=0),axis=0)
                neigh_degree_vec = np.append(neigh_degree_vec, np.expand_dims(int_to_multihot(degree_info[neighbor_info[vertices_id[i]][j]], vec_dim), axis=0),axis=0)
        # print(neigh_label_vec)
        neigh_label_vec = np.mean(neigh_label_vec, axis=0)
        neigh_degree_vec = np.mean(neigh_degree_vec, axis=0)
        # print(neigh_label_vec)
        current_feat = np.concatenate((label_vec, degree_vec, neigh_label_vec, neigh_degree_vec), axis=0)
        # print(current_feat)
        if i == 0:
            feature_vec = np.expand_dims(current_feat, axis=0)
        else:
            feature_vec = np.append(feature_vec, np.expand_dims(current_feat, axis=0), axis=0)
    return torch.from_numpy(feature_vec).type(torch.FloatTensor)


def preprocess_data_edge(vertices, origin_edge_list):
    vertices_dict = dict()
    new_e_u = list()
    new_e_v = list()
    for i in range(len(vertices)):
        vertices_dict[vertices[i]] = i
    for i in range(len(origin_edge_list[0])):
        new_e_u.append(vertices_dict[origin_edge_list[0][i]])
        new_e_v.append(vertices_dict[origin_edge_list[1][i]])
    return [new_e_u, new_e_v]


def preprocess_query2data(sub_vertices, candidate_info):
    total_len = len(candidate_info)
    num_query_vertex = total_len/2
    vertices_dict = dict()
    new_e_u = list()
    new_e_v = list()
    for i in range(len(sub_vertices)):
        vertices_dict[sub_vertices[i]] = i + num_query_vertex
    for i in range(total_len):
        if i%2 == 1:
            candidate_list = candidate_info[i].split()
            query_vertex = i//2
            for data_vertex in candidate_list:
                data_vertex = int(data_vertex)
                if data_vertex in sub_vertices:
                    new_e_u.append(query_vertex)
                    new_e_v.append(vertices_dict[data_vertex])
            try:
                candidate_list = candidate_info[i+2].split()
                data_vertex = int(candidate_list[0])
                new_e_u.append(query_vertex)
                new_e_v.append(vertices_dict[data_vertex])
            except IndexError:
                continue
    
    return [new_e_u, new_e_v]





def save_params(file_position, args):
    with open(file_position, 'w') as f:
        f.write('input feat dim:' + str(args.in_feat) +'\n')
        f.write('hidden dim: '+str(args.hidden_dim)+ '\n')
        f.write('output dim: '+str(args.out_dim) +'\n')
        f.write('learning rate: '+str(args.learning_rate) +'\n')
        f.write('epochs: '+str(args.num_epoch) +'\n')
        f.write('model: '+str(args.model_name) +'\n')
        f.write('training ratio: ' + str(args.train_percent) +'\n')
        f.write('train method: '+ str(args.train_method) + '\n')
        f.write('sample method: '+str(args.sample_method)+'\n')
