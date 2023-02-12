import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np


def main():
    fix_seed = 1234
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='GraphSAD: A GNN-based model for Subsequence Anomaly Detection')

    # basic config
    parser.add_argument('--model', type=str, default='MCAD_with_variable_length',
                        help='select a model from: [MCAD, Adaptive_MCAD, MCAD_with_variable_length]')
    parser.add_argument('--is_train', type=bool, default=True, help='train or test')

    # data settings
    parser.add_argument('--data', type=str, default='UCR_aug', help='dataset type')
    # parser.add_argument('--root_path', type=str, default='./data/UCR_Anomaly_FullData/', help='root path of the data file')
    parser.add_argument('--root_path', type=str, default='./data/KDD_full_augment/', help='root path of the data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--anomaly_injection', action='store_true', default=False, help='anomaly injection flag')

    # subsequence and graph settings
    parser.add_argument('--step_len', type=int, default=1, help='the length of a minimum element')
    parser.add_argument('--max_order', type=int, default=4, help='xxx')
    parser.add_argument('--min_order', type=int, default=0, help='xxx')
    parser.add_argument('--default_order', type=int, default=2, help='xxx')

    parser.add_argument('--input_dim', type=int, default=1, help='time series dimension')
    parser.add_argument('--stride_factor', type=float, default=0.125, help='ratio of stride length to period length')
    parser.add_argument('--reference_factor', type=float, default=1., help='ratio of reference length to period length')
    parser.add_argument('--k', type=int, default=3, help='the number of knn neighbours in semantic graph')

    # model settings
    parser.add_argument('--lamb', type=float, default=1., help='the weight of the autoencoder regularization')
    parser.add_argument('--hidden_dim', type=int, default=128, help='dimension of model')
    parser.add_argument('--num_layer', type=int, default=2, help='the number of gnn layers per anomaly detection block')
    parser.add_argument('--num_hop', type=int, default=1, help='the number of hop in graph to compute context')

    # optimization settings
    parser.add_argument('--train_epochs', type=int, default=12, help='train epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
    parser.add_argument('--des', type=str, default='test', help='exp description')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    # UCR_data_list = [
    #                     '038_UCR_Anomaly_DISTORTEDLab2Cmac011215EPG2_5000_27862_27932.txt',
    #                     '030_UCR_Anomaly_DISTORTEDInternalBleeding19_3000_4187_4197.txt',
    #                     '039_UCR_Anomaly_DISTORTEDLab2Cmac011215EPG3_5000_16390_16420.txt',
    #                     '121_UCR_Anomaly_ECG3_15000_16000_16100.txt',
    #                     '127_UCR_Anomaly_GP711MarkerLFM5z1_5000_6168_6212.txt',
    #                     '166_UCR_Anomaly_apneaecg_10000_12240_12308.txt',
    #                     '173_UCR_Anomaly_insectEPG1_3000_7000_7030.txt',
    #                     '191_UCR_Anomaly_resperation9_38000_143411_143511.txt'
    #                 ]

    probs, scores, accs = [], [], []
    data_list= os.listdir(args.root_path)
    for data in data_list:
        if str(data)[:5] == 'label':
            continue
        
        # if str(data)[5:8] in ['014','017','018','087']:
        #     continue
        
        # if int(str(data)[5:8]) <= 18:
        #     continue

        args.data_path = data
        args.step_len = 10
        print('Args in experiment:')
        print(args)


        # setting record of experiments
        setting = '{}'.format(args.model)


        exp = Exp_Main(args)  # set experiments
        print('>>>>>>>start: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        _, test_prob, score, acc_1, acc_3, acc_5, acc_10 = exp.run(setting)
        accs.append([acc_1, acc_3, acc_5, acc_10])
        torch.cuda.empty_cache()

        probs.append(test_prob)
        scores.append(score)
        print(np.mean(scores))
        print(np.mean(accs,axis = 0))
    # args.data_path = UCR_data_list[7]
    # print('Args in experiment:')
    # print(args)


    # # setting record of experiments
    # setting = '{}'.format(args.model)


    # exp = Exp_Main(args)  # set experiments
    # print('>>>>>>>start: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    # exp.run(setting)
    # torch.cuda.empty_cache()
    print(np.mean(scores))
    print(np.mean(accs,axis = 0))

if __name__ == "__main__":
    main()