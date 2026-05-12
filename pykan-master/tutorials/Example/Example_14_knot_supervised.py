#!/usr/bin/env python
# coding: utf-8

# # Example 14: Knot supervised

# In[1]:

import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append("/media/data/gaolili/deepLearning_project/pykan-master/")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from kan import *
import copy
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

def train_acc():
    return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).float())

def test_acc():
    return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())


is_training = True
# torch.manual_seed(seed)
# np.random.seed(seed)

# Download data: https://colab.research.google.com/github/deepmind/mathematics_conjectures/blob/main/knot_theory.ipynb#scrollTo=l10N2ZbHu6Ob
df = pd.read_csv("/media/data/gaolili/deepLearning_project/mathematics_conjectures-main/data/knot_theory_invariants.csv")
df.keys()#####df.keys()[7]:longitudinal_translation   df.keys[8]:meridinal_translation_image  df.keys[9]:meridinal_translation_real

X = df[df.keys()[1:-1]].to_numpy()
#X = X[:,7:10]
Y = df[['signature']].to_numpy()

# normalize X
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean[np.newaxis,:])/X_std[np.newaxis,:]
input_normalier = [X_mean, X_std]

# normalize Y
max_signature = np.max(Y)
min_signature = np.min(Y)
Y = ((Y-min_signature)/2).astype(int)
n_class = int((max_signature-min_signature)/2+1)
output_normalier = [min_signature, 2]

#seeds = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]
# 保存训练过程中的最优模型（基于 test_acc）
save_dir = "/media/data/gaolili/deepLearning_project/pykan-master/tutorials/Example/model"
os.makedirs(save_dir, exist_ok=True)
seeds = [200]

        
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
    dataset['train_label'] = torch.from_numpy(Y[train_id_][:,0]).type(torch.long).to(device)
    dataset['test_input'] = torch.from_numpy(X[test_id_]).type(dtype).to(device)
    dataset['test_label'] = torch.from_numpy(Y[test_id_][:,0]).type(torch.long).to(device)

    best_ckpt = os.path.join(save_dir, f"kan_knot_seed{seed}")
    
        
    if is_training:

        model = KAN(width=[n_feature,1,n_class], grid=5, k=3, seed=seed, device=device)
        results = model.fit(dataset, lamb=0.005, batch=1024, steps=200, loss_fn = nn.CrossEntropyLoss(),
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
            model_best = KAN(width=[n_feature,1,n_class], grid=5, k=3, seed=seed, device=device)
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



    model_best.plot(scale=1.0, beta=0.2)

    features = list(df.keys()[1:-1])
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
    

    scores = model_best.feature_score

    top3_values, top3_indices = torch.topk(scores, k=3, dim=0, largest=True)
    print("Top 3 feature importance values:", top3_values.detach().cpu().numpy())
    print("Top 3 feature indices:", top3_indices.detach().cpu().numpy())

    # 将 top3 值与索引追加写入文本文件（每个 seed 一行）
    results_file = os.path.join(save_dir, 'top3_features_by_seed.txt')
    vals = top3_values.detach().cpu().numpy().flatten()
    idxs = top3_indices.detach().cpu().numpy().flatten()
    # 格式：seed \t val1,val2,val3 \t idx1,idx2,idx3\n
    with open(results_file, 'a') as f:
        f.write(f"{seed}\t{','.join([str(float(v)) for v in vals])}\t{','.join([str(int(i)) for i in idxs])}\n")
    

    y_pos = np.arange(len(features))
    # plt.bar(y_pos, scores.detach().cpu().numpy())
    # plt.xticks(y_pos, features, rotation=90)
    # plt.ylabel('feature importance')
    # plt.show()

    plt.figure(figsize=(max(6, n * 0.4), 4))
    plt.bar(y_pos, scores.detach().cpu().numpy())
    plt.xticks(y_pos, features, rotation=90)

    plt.ylabel('feature importance')
    plt.title('KAN Feature Importance for Knot Theory')
    plt.tight_layout()  # 自动调整布局
    #plt.show()
    # 保存特征重要性图
    feat_plot_path = "/media/data/gaolili/deepLearning_project/pykan-master/figures/feature_importance.png"
    plt.savefig(feat_plot_path, dpi=150)
    plt.close()
    print(f"特征重要性图已保存至：{feat_plot_path}")
