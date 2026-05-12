#!/usr/bin/env python
# coding: utf-8

# # Example 14: Knot supervised

# In[1]:

#import matplotlib
#matplotlib.use('Agg')
import sys
from functools import partial
sys.path.append("/media/data/gaolili/deepLearning_project/pykan-master/")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from kan import *
import copy
import os
import random
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(device)

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

dtype = torch.get_default_dtype()


df = pd.read_csv("/media/data/gaolili/deepLearning_project/mathematics_conjectures-main/data/knot_theory_invariants.csv")
df.keys()#####df.keys()[7]:longitudinal_translation   df.keys[8]:meridinal_translation_image  df.keys[9]:meridinal_translation_real

X = df[df.keys()[1:-1]].to_numpy()
X = X[:,7:10]
Y = df[['signature']].to_numpy()
Y_max = np.max(np.abs(Y))

# normalize X
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean[np.newaxis,:])/X_std[np.newaxis,:]
input_normalier = [X_mean, X_std]




def train_acc():
    pred_real = model(dataset['train_input']).view(-1)
    label_real = dataset['train_label'].view(-1)
    # 如果预测值与真实 Signature 距离小于 1，则认为找对了那个偶数
    return torch.mean((torch.abs(pred_real - label_real) < 1.0).float())

def test_acc():
    pred_real = model(dataset['test_input']).view(-1)
    label_real = dataset['test_label'].view(-1) 
    return torch.mean((torch.abs(pred_real - label_real) < 1.0).float())


is_training = True
# torch.manual_seed(seed)
# np.random.seed(seed)

# Download data: https://colab.research.google.com/github/deepmind/mathematics_conjectures/blob/main/knot_theory.ipynb#scrollTo=l10N2ZbHu6Ob




#seeds = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]
# 保存训练过程中的最优模型（基于 test_acc）
save_dir = "/media/data/gaolili/deepLearning_project/pykan-master/tutorials/Example/model"
os.makedirs(save_dir, exist_ok=True)
seeds = [20]

        
for seed in seeds:
    set_seed(seed=seed)
    dataset = {}
    num = X.shape[0]
    n_feature = X.shape[1]
    train_ratio = 0.8
    train_id_ = np.random.choice(num, int(num*train_ratio), replace=False)
    test_id_ = np.array(list(set(range(num))-set(train_id_)))

    dtype = torch.get_default_dtype()
    
    dataset['train_input'] = torch.from_numpy(X[train_id_]).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(Y[train_id_][:]).type(dtype).to(device)
    dataset['test_input'] = torch.from_numpy(X[test_id_]).type(dtype).to(device)
    dataset['test_label'] = torch.from_numpy(Y[test_id_][:]).type(dtype).to(device)

    best_ckpt = os.path.join(save_dir, f"kan_knot_seed{seed}")
    
        
    if is_training:

        model = KAN(width=[n_feature, 1], grid=5, k=3, seed=seed, device=device)
        # results = model.fit(dataset, lamb=0.01, batch=1024, steps=200, lr=0.1,
        #                     metrics=[train_acc_, test_acc_], 
        #                     display_metrics=['train_loss', 'reg','train_acc', 'test_acc'],
        #                     update_grid=False, save_best=True, save_best_key='test_acc', save_best_path=best_ckpt)######reg_metric='edge_forward_spline_n', update_grid=True
        print("Step 0: Adam Warmup...")
        # results = model.fit(dataset, opt='Adam', steps=100, lr=0.01, lamb=0.0, 
        #                     metrics=[train_acc, test_acc], display_metrics=['train_loss', 'test_acc'])
        # pred = model(dataset['test_input'])
        # print(f"预测值范围: {pred.min().item():.2f} 到 {pred.max().item():.2f}")
        # print(f"真实值范围: {dataset['test_label'].min().item()} 到 {dataset['test_label'].max().item()}")
        
        # model = model.refine(50)

        results = model.fit(dataset, lamb=0.005, batch=1024, steps=50, opt='LBFGS', 
                            metrics=[train_acc, test_acc], display_metrics=['train_loss', 'reg', 'train_acc', 'test_acc'],
                            update_grid=True, save_best=True, save_best_key='test_acc', save_best_path=best_ckpt)######reg_metric='edge_forward_spline_n', update_grid=True
        
        
        


        test_acc_best = np.max(results['test_acc'])
        print("test_acc_best", test_acc_best)

    # 加载并评估保存的最优模型（优先使用库提供的 loadckpt）
    try:
        model_best = KAN.loadckpt(best_ckpt)
        print(f"已从检查点加载模型: {best_ckpt}")
    except Exception as e:
        state_path = best_ckpt + '_state_dict.pth'
        if os.path.exists(state_path):
            model_best = KAN(width=[n_feature,1], grid=5, k=3, seed=seed, device=device)
            model_best.load_state_dict(torch.load(state_path, map_location=device))
            print(f"已从 state_dict 加载模型: {state_path}")
        else:
            print(f"无法加载最优模型，使用当前训练模型。错误: {e}")
            model_best = model

    model_best.to(device)
    model_best.eval()
    with torch.no_grad():
        pred_real = model_best(dataset['test_input']).view(-1)
        label_real = dataset['test_label'].view(-1) 
        loaded_test_acc = torch.mean((torch.abs(pred_real - label_real) < 1.0).float()).item()
        
    print(f"Loaded model test accuracy: {loaded_test_acc:.4f}")


    folder= "./figures"
    model_best.plot(scale=1.0, beta=0.2)
    
    plt.savefig(f'{folder}/activation_function.png', bbox_inches="tight", dpi=400)
    plt.show()
    
    # model_best.prune()
    # model_best.plot(scale=1.0, beta=0.2) # 观察一下哪些线变细消失了
    # plt.savefig(f'{folder}/activation_function_prune.png', bbox_inches="tight", dpi=400)
    # plt.show()
    # --- 第二步：自动寻找符号函数 ---
    # 我们给模型一个常用的数学函数库
    #lib = ['x', 'x^2', 'x^3', 'sin', 'cos', 'exp', 'log', 'tan', 'sqrt']

    # auto_symbolic 会遍历所有激活函数，寻找拟合度最高的符号
    # grid_num 是采样点数，越多越准
    model_best.auto_symbolic()

    # --- 第三步：微调并打印公式 ---
    # 锁定符号后，模型会变成一个纯解析公式，我们需要最后训练一次系数（只有常数项在变）
    print("\n--- 正在优化符号公式的系数 ---")
    model_best.fit(dataset, opt='LBFGS', steps=50, batch=1024,
                    metrics=[train_acc, test_acc], display_metrics=['train_loss', 'reg', 'train_acc', 'test_acc'],
                    update_grid=True, save_best=True, save_best_key='test_acc', save_best_path=best_ckpt)

    # 获取最终公式
    # [0][0] 表示从输入层到输出层的第 1 个输出神经元的公式
    formula = model_best.symbolic_formula()[0][0]

    print("\n" + "="*50)
    print(f"【发现的纽结 Signature 公式】:")
    print(formula)
    print("="*50)
