import torch
import sys
import os

# 设置 PYTHONPATH
sys.path.append("/home/gaolili/deepLearning_project/pykan-master/")

from physical_kan_V2.spline import B_batch_physical_shared
from physical_kan_V2.physical_action_multi import UnifiedSmoothActivationWithCorrection
import argparse

def test_curvature():
    # 模拟 args
    args_dict = {
        'physical_function': 'set',
        'activation_gains': False,
        'softmax_center': False,
        'clamp_method': 'hard',
        'clamp_min_learn': True,
        'mode': 'real'
    }
    args = argparse.Namespace(**args_dict)

    # 模拟 activation_config 所需的参数 (简化版)
    # 我们只需要一个能运行的 physical_basic
    class DummyActivation(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            # 返回线性函数 x
            return x

    physical_basic = DummyActivation()

    # 设置网格和输入
    G = 5
    k = 3
    in_dim = 1
    grid = torch.linspace(-1, 1, steps=G+1)[None, :].expand(in_dim, G+1)
    
    # 扩展网格 (手动模拟 extend_grid)
    h = (grid[:, [-1]] - grid[:, [0]]) / G
    for _ in range(k):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)

    x = torch.linspace(-0.5, 0.5, steps=10).reshape(-1, 1).requires_grad_(True)

    # 计算 k=1 (旧版逻辑) 的基函数
    # 注意：现在我们的 B_batch_physical_shared 已经支持 k 了
    # 如果我们传 k=1
    bases_k1 = B_batch_physical_shared(x, grid, physical_basic, k=1)
    
    # 计算 k=3 (新版逻辑) 的基函数
    bases_k3 = B_batch_physical_shared(x, grid, physical_basic, k=3)

    def get_laplacian(bases, x):
        # 模拟不同系数的情况
        coef = torch.randn(bases.shape[1], bases.shape[2])
        u = (bases[0] * coef).sum()
        grad = torch.autograd.grad(u, x, create_graph=True)[0]
        lap = torch.autograd.grad(grad.sum(), x, create_graph=True)[0]
        return lap

    lap_k1 = get_laplacian(bases_k1, x)
    lap_k3 = get_laplacian(bases_k3, x)

    print(f"k=1 Laplacian mean absolute value: {lap_k1.abs().mean().item():.2e}")
    print(f"k=3 Laplacian mean absolute value: {lap_k3.abs().mean().item():.2e}")

    if lap_k3.abs().mean() > lap_k1.abs().mean():
        print("Success: k=3 provides more curvature than k=1!")
    else:
        print("Failure: k=3 curvature is not higher than k=1.")

if __name__ == "__main__":
    test_curvature()
