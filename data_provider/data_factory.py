from data_provider.datautils import *

data_dict = {
    'UCR': load_data_UCR,
    'UCR_aug': load_data_UCR_aug
}


def data_provider(args):
    data = args.data
    load_data = data_dict[data]

    x, y, period, stride, train_mask, test_mask, neighbor_mask, all_mask, num_nodes, origin_y = load_data(args)
    x_norm = (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + 1e-5)
    # x_norm = x[:,:period].copy()
    dist, idx = find_neighbors(x_norm, neighbor_mask=all_mask, k=args.k, length = 2 ** args.default_order * args.step_len)
    semantic_data = build_semantic_graph(x, y, dist, idx, args)
    # temporal_data = build_temporal_graph(x, y, period, stride)

    return semantic_data, train_mask, test_mask, all_mask, stride, num_nodes, period, origin_y

