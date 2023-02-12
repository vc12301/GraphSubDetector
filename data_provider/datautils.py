import pandas as pd
import numpy as np
import random
import tsaug
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf
from sklearn.metrics.pairwise import euclidean_distances
import faiss
import torch
from torch_geometric.data import Data


def load_data_UCR(args):

    root_path = args.root_path
    data_path = args.data_path
    
    max_order = args.max_order
    min_order = args.min_order
    default_order =args.default_order

    stride_factor = args.stride_factor
    reference_factor = args.reference_factor
    anomaly_injection_flag = args.anomaly_injection

    List = data_path[:-4].split('_')

    category = List[3]
    safe_num = int(List[-3])  # index of anomaly free sequence
    anomaly1 = int(List[-2])  # start of anomaly
    anomaly2 = int(List[-1])  # end of anomaly

    

    if pd.read_csv(root_path + data_path, header=None).iloc[:, 0].dtype == 'O':
        sr = pd.read_csv(root_path + data_path, header=None, sep=r"\s+").T.iloc[:, 0]
    else:
        sr = pd.read_csv(root_path + data_path, header=None).iloc[:, 0]
    # sr = pd.read_csv(root_path + data_path, header=None).iloc[:, 0]
    print(f'The total length is {sr.shape[0]}, the anomaly is between {anomaly1} and {anomaly2}')
    
    origin_y = np.ones(sr.shape[0])
    origin_y[anomaly1:anomaly2+1] = 1.

    period = get_period(sr)
    args.step_len = max(int(0.125 * period), args.step_len)
    step_len = args.step_len

    print(f'The period length is {period}.')
    stride = int(period * stride_factor)
    stride = max(stride, 1)
    stride = max(stride, int(sr.shape[0]/7500))
    reference_len = int(period * reference_factor)
    max_len = 2 ** max_order * step_len
    min_len = 2 ** min_order * step_len
    default_len = 2 ** default_order * step_len
    print(f'The stride is {stride}, and the step length is {step_len}.')
    print(f'The max length is {max_len}, the min length is {min_len}, and the default length is {default_len}.')

    if anomaly_injection_flag:
        sr, training_anomalies = anomaly_injection(sr.values, safe_num, period)

    #build graph
    max_win_size = max_len
    sr = normalizing_series(sr, mode='zero-score')
    num_nodes = int((len(sr)-max_win_size)/stride) + 1
    #generate subsequences
    subseq = np.array([ sr[k*stride : k*stride+max_win_size ] for k in range(num_nodes)] )
    subseq = subseq.astype('float32')
    labels = np.ones(num_nodes) # nomimal label
    #calculate the euclidean distance matries

    train_len = int((safe_num - max_win_size + 1) / stride)
    test_len = num_nodes - train_len

    # labeling anomalies in test set
    for k in range(num_nodes):
        s = k * stride
        e = k * stride + reference_len
        if (anomaly1 <= s <= anomaly2) or (anomaly1 <= e <= anomaly2) or (s <= anomaly1 <= anomaly2 <=e) :
            labels[k] = 0. # anomaly label
    print('The number of samples: ', num_nodes)
    print('The number of training samples: ', train_len)
    print('The number of anomaly samples: ', int(num_nodes - labels.sum()))

    # inject anomalies in train set
    if anomaly_injection_flag:
        for k in range(num_nodes):
            s = k * stride
            e = k * stride + max_win_size
            for anomaly_s, anomaly_e in training_anomalies:
                if (anomaly_s <= s <= anomaly_e) or (anomaly_s <= e <= anomaly_e) or (s <= anomaly_s <= anomaly_e <=e) :
                    labels[k] = 0. # anomaly label
    print('The number of anomaly samples after anomalies injection: ', int(num_nodes - labels.sum()))


    train_mask = np.hstack((np.ones(train_len),np.zeros(test_len)))
    test_mask = np.hstack((np.zeros(train_len),np.ones(test_len)))
    neighbor_mask = np.hstack((np.ones(train_len),np.zeros(test_len)))
    all_mask = np.ones(num_nodes)

    return subseq, labels, period, stride, train_mask, test_mask, neighbor_mask, all_mask, num_nodes, origin_y



def load_data_UCR_aug(args):

    root_path = args.root_path
    data_path = args.data_path
    
    max_order = args.max_order
    min_order = args.min_order
    default_order =args.default_order

    stride_factor = args.stride_factor
    reference_factor = args.reference_factor
    anomaly_injection_flag = args.anomaly_injection

    List = data_path[:-4].split('_')
    safe_num = int(List[-1])

    if pd.read_csv(root_path + data_path, header=None).iloc[:, 0].dtype == 'O':
        sr = pd.read_csv(root_path + data_path, header=None, sep=r"\s+").T.iloc[:, 0]
    else:
        sr = pd.read_csv(root_path + data_path, header=None).iloc[:, 0]

    period = get_period(sr)
    args.step_len = max(int(0.125 * period), args.step_len)
    step_len = args.step_len

    print(f'The period length is {period}.')
    stride = int(period * stride_factor)
    stride = max(stride, 1)
    stride = max(stride, int(sr.shape[0]/7500))
    reference_len = int(period * reference_factor)
    max_len = 2 ** max_order * step_len
    min_len = 2 ** min_order * step_len
    default_len = 2 ** default_order * step_len
    print(f'The stride is {stride}, and the step length is {step_len}.')
    print(f'The max length is {max_len}, the min length is {min_len}, and the default length is {default_len}.')

    if anomaly_injection_flag:
        sr, training_anomalies = anomaly_injection(sr.values, safe_num, period)

    #build graph
    max_win_size = max_len
    sr = normalizing_series(sr, mode='zero-score')
    num_nodes = int((len(sr)-max_win_size)/stride) + 1
    #generate subsequences
    subseq = np.array([ sr[k*stride : k*stride+max_win_size ] for k in range(num_nodes)] )
    subseq = subseq.astype('float32')
    labels = np.ones(num_nodes) # nomimal label
    #calculate the euclidean distance matries

    train_len = int((safe_num - max_win_size + 1) / stride)
    test_len = num_nodes - train_len

    flabel = pd.read_csv(root_path + 'label' + data_path[4:], header=None).iloc[:, 0]

    for k in range(num_nodes):
        s = k * stride
        e = k * stride + reference_len
        if sum(flabel[s:e]) > 1:
            labels[k] = 0 # nomimal label

    print('The number of samples: ', num_nodes)
    print('The number of training samples: ', train_len)
    print('The number of anomaly samples: ', int(num_nodes - labels.sum()))

    # inject anomalies in train set
    if anomaly_injection_flag:
        for k in range(num_nodes):
            s = k * stride
            e = k * stride + max_win_size
            for anomaly_s, anomaly_e in training_anomalies:
                if (anomaly_s <= s <= anomaly_e) or (anomaly_s <= e <= anomaly_e) or (s <= anomaly_s <= anomaly_e <=e) :
                    labels[k] = 0. # anomaly label
    print('The number of anomaly samples after anomalies injection: ', int(num_nodes - labels.sum()))

    train_mask = np.hstack((np.ones(train_len), np.zeros(test_len)))
    test_mask = np.hstack((np.zeros(train_len), np.ones(test_len)))
    neighbor_mask = np.hstack((np.ones(train_len), np.zeros(test_len)))
    all_mask = np.ones(num_nodes)

    return subseq, labels, period, stride, train_mask, test_mask, neighbor_mask, all_mask, num_nodes, flabel


def normalize_dis(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = np.diag(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def get_period(input_data):
    #     input_data, trend_signal = sm.tsa.filters.hpfilter(input_data, lamb=1e8)
    diff_acf = acf(input_data, nlags=len(input_data), fft=False)
    all_peaks, _ = find_peaks(diff_acf, height=0.01)
    peak_th = np.mean(diff_acf[all_peaks])

    peaks_diff_data, _ = find_peaks(diff_acf, height=peak_th / 2, prominence=peak_th / 4)
    #     default = 100
    if len(peaks_diff_data) <= 2:
        return 100

    peaks_2nd_diff = (np.diff(np.hstack((0, peaks_diff_data))))

    win_size = int(np.median(peaks_2nd_diff))

    return win_size

def normalizing_series(ts, mode='zero-score'):
    if mode == 'min-max':
        ts_max = max(ts)
        ts_min = min(ts)
        return (ts - ts_min) / (ts_max - ts_min)
    elif mode == 'zero-score':
        mu = ts.mean()
        sigma = ts.std()
        return (ts - mu) / (sigma + 1e-5)
    else:
        raise ValueError(f'Unknown normalization mode: {mode}')


def find_neighbors(x, neighbor_mask, k, length):
    # nearest neighbour object
    index = faiss.IndexFlatL2(x.shape[-1])
    # add nearest neighbour candidates
    index.add(x)
    # distances and idx of neighbour points for the neighbour candidates (k+1 as the first one will be the point itself)
    dist, idx = index.search(x, k=k+1)
    # remove 1st nearest neighbours to remove self loops
    dist, idx = dist[:, 1:] / length, idx[:, 1:]
    return dist, idx


def build_semantic_graph(x, y, dist, idx, args, sigma=10.):

    max_order = args.max_order
    min_order = args.min_order
    step_len = args.step_len

    # array like [0,0,0,0,0,1,1,1,1,1,...,n,n,n,n,n] for k = 5 (i.e. edges sources)
    idx_source = np.repeat(np.arange(len(x)), dist.shape[-1]).astype('int32')
    # edge targets, i.e. the nearest k neighbours of point 0, 1,..., n
    idx_target = idx.flatten()

    dists = []
    for order in range(min_order, max_order+1):
        length = 2 ** order * step_len
        euclidean_dist = np.linalg.norm(x[idx_source,:length] - x[idx_target,:length], ord=2, axis=1)
        x_source, x_target = x[idx_source,:length].copy(), x[idx_target,:length].copy()
        x_source = (x_source - np.mean(x_source, axis=-1, keepdims=True)) / (np.std(x_source, axis=-1, keepdims=True) + 1e-5)
        x_target = (x_source - np.mean(x_target, axis=-1, keepdims=True)) / (np.std(x_target, axis=-1, keepdims=True) + 1e-5)
        z_score_euclidean_dist = np.linalg.norm(x_source - x_target, ord=2, axis=1)
        dists.append(euclidean_dist)
        dists.append(z_score_euclidean_dist)
    attr = np.stack(dists, axis=-1)
    std = np.std(attr, axis=0, keepdims=True)
    attr = np.exp(-attr / (sigma * std))

    idx_source = np.expand_dims(idx_source, axis=0)
    idx_target = np.expand_dims(idx_target, axis=0).astype('int32')

    # stack source and target indices, and reverse them
    idx = np.vstack((idx_target, idx_source))

    # # edge weights, distance to similairty
    # attr = dist.flatten()
    # attr = np.exp(-attr / sigma)
    # # attr = np.sqrt(attr)
    # attr = np.expand_dims(attr, axis=1)

    # into tensors
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    idx = torch.tensor(idx, dtype=torch.long)
    attr = torch.tensor(attr, dtype=torch.float32)
    # build PyTorch geometric Data object
    data = Data(x=x, edge_index=idx, edge_attr=attr, y=y)

    return data


# def build_temporal_graph(x, y, period, stride):

#     node_idx = np.arange(len(x)).astype('int32')
#     self_connection_idx = np.vstack((node_idx, node_idx))

#     idx_source = np.arange(len(x)-1).astype('int32')
#     idx_target = np.arange(1,len(x)).astype('int32')
#     temporal_edge_idx = np.concatenate((np.vstack((idx_source, idx_target)), np.vstack((idx_target, idx_source))), axis=-1)
#     # temporal_edge_idx = np.vstack((idx_target, idx_source))

#     step = round(period / stride)
#     if step > 1:
#         idx_source = np.arange(len(x)-step).astype('int32')
#         idx_target = np.arange(step, len(x)).astype('int32')
#         periodic_edge_idx = np.concatenate((np.vstack((idx_source, idx_target)), np.vstack((idx_target, idx_source))), axis=-1)
#         # periodic_edge_idx = np.vstack((idx_target, idx_source))
#         idx = np.concatenate((temporal_edge_idx, periodic_edge_idx), axis=-1)
#     else:
#         idx = temporal_edge_idx

#     attr = np.ones((idx.shape[-1],1))

#     # into tensors
#     x = torch.tensor(x, dtype=torch.float32)
#     y = torch.tensor(y, dtype=torch.float32)
#     idx = torch.tensor(idx, dtype=torch.long)
#     attr = torch.tensor(attr, dtype=torch.float32)
#     data = Data(x=x, edge_index=idx, edge_attr=attr, y=y)
#     return data

def build_temporal_graph(x, y, period, stride,num_stride=5):

    node_idx = np.arange(len(x)).astype('int32')
    self_connection_idx = np.vstack((node_idx, node_idx))
    self_connection_att = np.ones((self_connection_idx.shape[-1],1))

    idx_source = np.arange(len(x)-1).astype('int32')
    idx_target = np.arange(1,len(x)).astype('int32')
    temporal_edge_idx = np.concatenate((np.vstack((idx_source, idx_target)), np.vstack((idx_target, idx_source))), axis=-1)
    temporal_edge_att = np.exp(-np.ones((temporal_edge_idx.shape[-1],1)))

    step = round(period / stride)
    if step> 1:
        num_stride = min(step,num_stride)
        for i in range(1, num_stride-1):
            idx_source = np.arange(len(x) - i-1).astype('int32')
            idx_target = np.arange(i+1, len(x)).astype('int32')
            tmp_idx = np.concatenate((np.vstack((idx_source, idx_target)), np.vstack((idx_target, idx_source))), axis=-1)
            tmp_att = np.exp(-np.ones((tmp_idx.shape[-1],1))*(i+1))
            temporal_edge_idx = np.concatenate((temporal_edge_idx,tmp_idx),axis = -1)
            temporal_edge_att = np.concatenate((temporal_edge_att,tmp_att),axis = 0)



    idx_source = np.arange(len(x)-step).astype('int32')
    idx_target = np.arange(step, len(x)).astype('int32')
    periodic_edge_idx = np.concatenate((np.vstack((idx_source, idx_target)), np.vstack((idx_target, idx_source))), axis=-1)
    periodic_edge_att = np.ones((periodic_edge_idx.shape[-1],1))

    idx = np.concatenate((self_connection_idx, temporal_edge_idx, periodic_edge_idx), axis=-1)
    attr = np.concatenate((self_connection_att, temporal_edge_att, periodic_edge_att), axis=0)

    # into tensors
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    idx = torch.tensor(idx, dtype=torch.long)
    attr = torch.tensor(attr, dtype=torch.float32)
    data = Data(x=x, edge_index=idx, edge_attr=attr, y=y)
    return data


def anomaly_injection(sr, train_end, period, num=1):
    anomaly_types = [
        'spike or dip',
        'reverse_left_right',
        'reverse_up_down',
        'resize',
        'warping',
        'noise_injection'
        ]
    
    anomalies = []
    for i in range(num):

        anomaly_type = anomaly_types[i%5]

        anomaly_len = int(random.uniform(0.5, 1.2) * period)
        if anomaly_type == 'spike or dip':
            anomaly_len = 1
        start = random.randint(0, train_end-anomaly_len)
        end = start + anomaly_len
        
        mean, std = sr.mean(), sr.std()
        if anomaly_type == 'spike or dip':
            seed = random.randint(0, 1)
            if seed == 0:
                sr[start] += random.uniform(0.5, 2.0) * std
            else:
                sr[start] -= random.uniform(0.5, 2.0) * std
            
        elif anomaly_type == 'reverse_left_right':
            sr[start:end] = sr[start:end][::-1]

        elif anomaly_type == 'reverse_up_down':
            mean = sr[start:end].mean()
            sr[start:end] = -sr[start:end] + 2 * mean

        elif anomaly_type == 'resize':
            sr_temp = sr[start:end].copy()
            seed = random.randint(0, 1)
            if seed == 0:
                resize_len = int(random.uniform(1.25, 3.0) * anomaly_len)
            if seed == 1:
                resize_len = int(random.uniform(0.33, 0.8) * anomaly_len)
            sr_temp = tsaug.Resize(size=resize_len).augment(sr_temp)
            end = start + resize_len
            sr[start:end] = sr_temp

        elif anomaly_type == 'warping':
            sr_temp = sr[start:end].copy()
            sr_temp = tsaug.TimeWarp(n_speed_change=3, max_speed_ratio=3).augment(sr_temp)
            sr[start:end] = sr_temp

        elif anomaly_type == 'noise_injection':
            sr_temp = sr[start:end].copy()
            sr_temp = tsaug.AddNoise(scale=random.uniform(0.1, 0.3)*std).augment(sr_temp)
            sr[start:end] = sr_temp
        
        anomalies.append((start, end))

    return sr, anomalies
        