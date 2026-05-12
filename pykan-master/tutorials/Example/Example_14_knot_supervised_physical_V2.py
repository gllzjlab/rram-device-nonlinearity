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
import random
import torch
import sys
sys.path.append("/media/data/gaolili/deepLearning_project/pykan-master/")
from physical_kan_V2 import *
import copy
from physical_kan_V2.utils_physical import get_evaluate_extended_model_torch_params
dtype = torch.get_default_dtype()

def train_acc():
    return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).float())

def test_acc():
    return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 下面两行是关键，但也可能让训练变慢
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 强制确定性算法
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    parser = argparse.ArgumentParser(description="Train knot theory.")
    parser.add_argument("-mode", type=str, default='real', help="real/reco")
    parser.add_argument("-coef_negtive", action='store_true', default=False, help="activation weight coeff(True/False)")
    parser.add_argument('-clamp_min_default', type=float, default=-6.0)
    parser.add_argument('-clamp_max_default', type=float, default=6.0)
    parser.add_argument('-clamp_min_residual', type=float, default=-6.0)  # 残差连接后更严格
    parser.add_argument('-clamp_max_residual', type=float, default=6.0)
    parser.add_argument('-init_low', type=float, default=-1.0)
    parser.add_argument('-init_high', type=float, default=1.0)
    parser.add_argument("-input_dim", type=int, default=17, help="17/3")
    parser.add_argument("-input_dim_name", type=str, default='meridinal_translation_real', help="'meridinal_translation_image'/'meridinal_translation_real'")
    parser.add_argument("-lr", type=float, default=1.0, help="lr")
    parser.add_argument("-scheduler", action='store_true', default=False, help="adjust lr")
    parser.add_argument("-epochs", type=int, default=200, help="epoch")
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


    is_training = True
    
    input_dim = args.input_dim
    # Download data: https://colab.research.google.com/github/deepmind/mathematics_conjectures/blob/main/knot_theory.ipynb#scrollTo=l10N2ZbHu6Ob
    df = pd.read_csv(
        "/media/data/gaolili/deepLearning_project/mathematics_conjectures-main/data/knot_theory_invariants.csv")
    df.keys()

    X = df[df.keys()[1:-1]].to_numpy()
    if args.input_dim == 3:
        X = X[:,7:10]

    if args.input_dim == 1 and args.input_dim_name == 'meridinal_translation_image':
        X = X[:,8]
        X = X[:, np.newaxis]
    if args.input_dim == 1 and args.input_dim_name == 'meridinal_translation_real':
        X = X[:,9]
        X = X[:, np.newaxis]
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

    #seeds = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    seeds = [20]
    # 保存训练过程中的最优模型（基于 test_acc）
    save_ckeckpoint_dir = "./tutorials/Example/model_physical"
    os.makedirs(save_ckeckpoint_dir, exist_ok=True)
         
    for seed in seeds:
        set_seed(seed=seed)
        dataset = {}
        num = X.shape[0]
        n_feature = X.shape[1]
        train_ratio = 0.8
        train_id_ = np.random.choice(num, int(num * train_ratio), replace=False)
        test_id_ = np.array(list(set(range(num)) - set(train_id_)))

        
        dataset['train_input'] = torch.from_numpy(X[train_id_]).type(dtype).to(device)
        dataset['train_label'] = torch.from_numpy(Y[train_id_][:, 0]).type(torch.long).to(device)
        dataset['test_input'] = torch.from_numpy(X[test_id_]).type(dtype).to(device)
        dataset['test_label'] = torch.from_numpy(Y[test_id_][:, 0]).type(torch.long).to(device)

        best_ckpt = os.path.join(save_ckeckpoint_dir, f"kan_knot_seed{seed}_input_dim{input_dim}")

        if is_training:

            model = KAN(width=[n_feature,1,n_class], grid=5, is_extend_grid = False, seed=seed, device=device, coef_negative=args.coef_negtive, activation_config = activation_config)

            results = model.fit(dataset, lamb=0.0005, opt="LBFGS", lr=args.lr, batch=1024, steps=args.epochs, 
                                loss_fn = nn.CrossEntropyLoss(), metrics=[train_acc, test_acc], 
                                display_metrics=['train_loss', 'reg', 'train_acc', 'test_acc'], update_grid=False,
                                save_best=True, save_best_key='test_acc', save_best_path=best_ckpt)
            
            test_acc_best = np.max(results['test_acc'])
            print("test_acc_best", test_acc_best)
            # In[ ]:

            # 如果任一 coef 含有 NaN，则放弃该 seed
            coef0 = model.act_fun[0].coef
            coef1 = model.act_fun[1].coef
            print("coef_0", coef0)
            print("coef_1", coef1)
            
            if torch.isnan(coef0).any() or torch.isnan(coef1).any():
                print(f"Detected NaN in coefficients for seed {seed}; skipping this seed.")
                continue

        # 加载并评估保存的最优模型（优先使用库提供的 loadckpt）
        try:
            model_best = KAN.loadckpt(path=best_ckpt, activation_config=activation_config)
            print(f"已从检查点加载模型: {best_ckpt}")
        except Exception as e:
            state_path = best_ckpt + '_state_dict.pth'
            if os.path.exists(state_path):
                                          
                model_best = KAN(width=[n_feature,1,n_class], grid=5, is_extend_grid = False, seed=seed, device=device, coef_negative=args.coef_negtive, activation_config = activation_config)
                model_best.load_state_dict(torch.load(state_path, map_location=device))
                print(f"已从 state_dict 加载模型: {state_path}")
                    
            else:
                print(f"无法加载最优模型，使用当前训练模型。错误: {e}")
                model_best = model

        model_best.to(device)
        model_best.eval()
        with torch.no_grad():
            preds = torch.argmax(model_best(dataset['test_input']), dim=1)
            loaded_test_acc = torch.mean((preds == dataset['test_label']).float()).item()
            print(f"Loaded model test accuracy: {loaded_test_acc:.4f}")

        save_path = './physical_figures'
        os.makedirs(save_path, exist_ok=True)

        model.plot(scale=1.0, beta=0.8, folder=save_path, save_name='activation_function_seed_'+str(seed)+'_input_dim_'+str(input_dim)+'.png')
        
        if input_dim == 17:
            features = list(df.columns[1:-1])
            n = len(features)
            
            axes = plt.gcf().get_axes()
            if len(axes) > 0:
                ax = axes[0]
            else:
                ax = plt.gca()
            for i, name in enumerate(features):
                x = (i + 0.5) / n
                ax.text(x, -0.06, name, rotation=270, rotation_mode="anchor",
                        transform=ax.transAxes, ha='center', va='top')


            # In[ ]:

            scores = model.feature_score
            top3_values, top3_indices = torch.topk(scores, k=3, dim=0, largest=True)
            print("Top 3 feature importance values:", top3_values.detach().cpu().numpy())
            print("Top 3 feature indices:", top3_indices.detach().cpu().numpy())

            # 将 top3 值与索引追加写入文本文件（每个 seed 一行）
            # 每个 seed 使用单独的文件，覆盖写入
            results_file = os.path.join(save_path, f'top3_features_seed{seed}.txt')
            vals = top3_values.detach().cpu().numpy().flatten()
            idxs = top3_indices.detach().cpu().numpy().flatten()
            # 格式：seed \t val1,val2,val3 \t idx1,idx2,idx3\n
            # 将 test_acc_best（若存在）或 loaded_test_acc 写入文件，方便记录训练/加载结果
            try:
                acc_val = float(test_acc_best)
            except Exception:
                try:
                    acc_val = float(loaded_test_acc)
                except Exception:
                    acc_val = 'NA'

            with open(results_file, 'a') as f:
                f.write(f"{seed}\t{acc_val}\t{','.join([str(float(v)) for v in vals])}\t{','.join([str(int(i)) for i in idxs])}\n")
            
            

            ###########保存输入重要性特征图##############
            y_pos = np.arange(len(features))
            # plt.bar(y_pos, scores)
            # plt.xticks(y_pos, features, rotation=90);
            # plt.ylabel('feature importance')
            
            
            plt.figure(figsize=(max(6, n * 0.4), 4))
            plt.bar(y_pos, scores.detach().cpu().numpy())
            plt.xticks(y_pos, features, rotation=90)
            plt.ylabel('feature importance')
            plt.title('Physical KAN Feature Importance for Knot Theory')
            plt.tight_layout()  # 自动调整布局
            # plt.show()
            # 保存特征重要性图
            feat_plot_path = os.path.join(save_path, 'feature_importance_physical_seed_'+str(seed)+'_input_dim_'+str(input_dim)+'.png')
            plt.savefig(feat_plot_path, dpi=150)
            plt.close()
            print(f"特征重要性图已保存至：{feat_plot_path}")
