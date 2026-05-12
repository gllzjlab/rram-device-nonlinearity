import torch
import sys
import os
import numpy as np

# 设置 PYTHONPATH
sys.path.append("/home/gaolili/deepLearning_project/pykan-master/")

from physical_kan_V2.spline import B_batch_physical_shared, B_batch_physical_shared_C2

def test_c2_smoothness():
    class DummyActivation(torch.nn.Module):
        def forward(self, x):
            # 物理部分是线性的，整体平滑度取决于网格基函数
            return x

    physical_basic = DummyActivation()
    
    # 设置网格
    G = 5
    in_dim = 1
    grid = torch.linspace(-1, 1, steps=G+1)[None, :].expand(in_dim, G+1)
    
    # 在网格点附近进行高密度采样
    x = torch.linspace(-0.01, 0.01, steps=1000).reshape(-1, 1).requires_grad_(True)

    def get_derivatives(basis_func):
        bases = basis_func(x, grid, physical_basic)
        # 模拟不同系数的情况，形状与 bases 的后两个维度匹配
        coef = torch.randn(bases.shape[1], bases.shape[2])
        y = (bases[0] * coef).sum()
        
        # 一阶导数
        dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
        # 二阶导数
        d2y_dx2 = torch.autograd.grad(dy_dx.sum(), x, create_graph=True)[0]
        return dy_dx.detach(), d2y_dx2.detach()

    import matplotlib.pyplot as plt
    
    dy_linear, d2y_linear = get_derivatives(B_batch_physical_shared)
    dy_c2, d2y_c2 = get_derivatives(B_batch_physical_shared_C2)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x.detach().numpy(), dy_linear.numpy(), label='Linear (C0)')
    plt.plot(x.detach().numpy(), dy_c2.numpy(), label='C2 Smooth')
    plt.title("First Derivative (dy/dx)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x.detach().numpy(), d2y_linear.numpy(), label='Linear (C0)')
    plt.plot(x.detach().numpy(), d2y_c2.numpy(), label='C2 Smooth')
    plt.title("Second Derivative (d2y/dx2)")
    plt.legend()
    
    plt.savefig("/home/gaolili/deepLearning_project/pykan-master/smoothness_comparison.png")
    print("Plot saved to /home/gaolili/deepLearning_project/pykan-master/smoothness_comparison.png")

if __name__ == "__main__":
    test_c2_smoothness()
