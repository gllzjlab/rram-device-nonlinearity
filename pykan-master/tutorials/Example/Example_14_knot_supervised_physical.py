#!/usr/bin/env python
# coding: utf-8

# # Example 14: Knot supervised

# In[1]:

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import numpy as np
import torch

from physical_kan import *
import copy
from physical_kan.utils_physical import get_evaluate_extended_model_torch_params


def train_acc():
    return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).float())

def test_acc():
    return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    seed = 20#20
    torch.manual_seed(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser(description="Train mnist.")
    parser.add_argument("-activation_name", type=str, default='relu', help="relu/relu_bn/hybrid")
    parser.add_argument("-mode", type=str, default='real', help="real/reco")
    parser.add_argument("-coef_negtive", action='store_true', default=False, help="activation weight coeff(True/False)")
    parser.add_argument("-dataset", type=str, default='mnist', help="mnist/cifar10")
    parser.add_argument('-clamp_min_default', type=float, default=-6.0)
    parser.add_argument('-clamp_max_default', type=float, default=6.0)
    parser.add_argument('-clamp_min_residual', type=float, default=-6.0)  # 残差连接后更严格
    parser.add_argument('-clamp_max_residual', type=float, default=6.0)
    parser.add_argument('-init_low', type=float, default=-6.0)
    parser.add_argument('-init_high', type=float, default=6.0)
    parser.add_argument("-lr", type=float, default=1e-3, help="lr")
    parser.add_argument("-scheduler", action='store_true', default=False, help="adjust lr")
    parser.add_argument("-epochs", type=int, default=50, help="epoch")
    parser.add_argument("-trial_start", type=int, default=1, help="epoch")
    parser.add_argument("-trial_end", type=int, default=6, help="epoch")
    parser.add_argument("-clamp_method", type=str, default='hard', help="hard/software/scale_bias")
    parser.add_argument("-Nonlinear_method", type=str, default='shared', help="shared/no_shared")
    parser.add_argument("-clamp_min_learn", action='store_true', default=True, help="activation input clamp_min or learn(True/False)")
    parser.add_argument("-physical_function", type=str, default='all', help="set/reset/all")
    parser.add_argument("-activation_gains", action='store_true', default=False,
                        help="physical_function in activation gains(True/False)")
    parser.add_argument("-softmax_center", action='store_true', default=False,
                        help="softmax weight(True/False)")
    args = parser.parse_args()


    op = 'set'
    mode = args.mode
    path = '/media/data/gaolili/deepLearning_project/pykan_physics_function_V2/data/conduct_function_fit_result'
    breakpoints_set, model_funcs_set, params_set, value_domain_set, corrections_set, bound_values_set = get_evaluate_extended_model_torch_params(
        mode=mode, op=op, path=path)
    op = 'reset'
    breakpoints_reset, model_funcs_reset, params_reset, value_domain_reset, corrections_reset, bound_values_reset = get_evaluate_extended_model_torch_params(
        mode=mode, op=op, path=path)

    activation_config = {
                'args': args,
                'breakpoints_set': breakpoints_set,
                'model_funcs_set': model_funcs_set,
                'params_set': params_set,
                'value_domain_set': value_domain_set,
                'corrections_set': corrections_set,
                'bound_values_set': bound_values_set,
                'breakpoints_reset': breakpoints_reset,
                'model_funcs_reset': model_funcs_reset,
                'params_reset': params_reset,
                'value_domain_reset': value_domain_reset,
                'corrections_reset': corrections_reset,
                'bound_values_reset': bound_values_reset,
                'init_low': args.init_low,
                'init_high': args.init_high,
            }



    # Download data: https://colab.research.google.com/github/deepmind/mathematics_conjectures/blob/main/knot_theory.ipynb#scrollTo=l10N2ZbHu6Ob
    df = pd.read_csv(
        "/media/data/gaolili/deepLearning_project/mathematics_conjectures-main/data/knot_theory_invariants.csv")
    df.keys()

    X = df[df.keys()[1:-1]].to_numpy()
    Y = df[['signature']].to_numpy()

    # normalize X
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean[np.newaxis, :]) / X_std[np.newaxis, :]
    input_normalier = [X_mean, X_std]

    # normalize Y
    max_signature = np.max(Y)
    min_signature = np.min(Y)
    Y = ((Y - min_signature) / 2).astype(int)
    n_class = int((max_signature - min_signature) / 2 + 1)
    output_normalier = [min_signature, 2]

    dataset = {}
    num = X.shape[0]
    n_feature = X.shape[1]
    train_ratio = 0.8
    train_id_ = np.random.choice(num, int(num * train_ratio), replace=False)
    test_id_ = np.array(list(set(range(num)) - set(train_id_)))

    dtype = torch.get_default_dtype()
    dataset['train_input'] = torch.from_numpy(X[train_id_]).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(Y[train_id_][:, 0]).type(torch.long).to(device)
    dataset['test_input'] = torch.from_numpy(X[test_id_]).type(dtype).to(device)
    dataset['test_label'] = torch.from_numpy(Y[test_id_][:, 0]).type(torch.long).to(device)



    model = KAN(width=[n_feature,2,n_class], seed=seed, device=device, coef_negative=args.coef_negtive, activation_config = activation_config)

    results = model.fit(dataset, lamb=0.001, opt="LBFGS", lr=1.0, batch=1024, steps=200, loss_fn = nn.CrossEntropyLoss(), metrics=[train_acc, test_acc], display_metrics=['train_loss', 'reg', 'train_acc', 'test_acc'], update_grid=False)
    test_acc_best = np.max(results['test_acc'])
    print("test_acc_best", test_acc_best)
    # In[ ]:


    model.plot(scale=1.0, beta=0.2)

    n = 17
    for i in range(n):
        plt.gcf().get_axes()[0].text(1/(2*n)+i/n-0.005,-0.02,df.keys()[1:-1][i], rotation=270, rotation_mode="anchor")


    # In[ ]:


    scores = model.feature_score
    features = list(df.keys()[1:-1])

    y_pos = range(len(features))
    # plt.bar(y_pos, scores)
    # plt.xticks(y_pos, features, rotation=90);
    # plt.ylabel('feature importance')
    #
    # plt.figure(figsize=(12, 6))
    # plt.bar(y_pos, scores.detach().cpu().numpy())
    # plt.xticks(y_pos, features, rotation=90)

    plt.ylabel('feature importance')
    plt.title('KAN Feature Importance for Knot Theory')
    plt.tight_layout()  # 自动调整布局
    # plt.show()
    # 保存特征重要性图
    feat_plot_path = "/media/data/gaolili/deepLearning_project/pykan-master/result_figures/feature_importance_physical.png"
    plt.savefig(feat_plot_path, dpi=150)
    plt.close()
    print(f"特征重要性图已保存至：{feat_plot_path}")