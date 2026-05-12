import torch
from torch import autograd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import argparse
import os

sys.path.append("/media/data/gaolili/deepLearning_project/pykan-master/")
from physical_kan_V2 import KAN, LBFGS, ex_round
from physical_kan_V2.utils_physical import get_evaluate_extended_model_torch_params

def main():
    # --- 破局参数配置 ---
    # 我们挑选一组最有潜力的组合，并加入 noise_scale 和 alpha 调整
    best_params = {
        'lr': 0.01,               # 大幅降低学习率，从 0.1 降到 0.01，防止 LBFGS 步子太大导致 NaN
        'hidden_neurons': 5,      # 适当减小神经元数量，降低模型初期不稳定性
        'grid_size': 5,
        'coef_negtive': False,
        'update_grid': True,
        'physical_function': 'set',
        'epochs': 200,
        'mode': 'real',
        'noise_scale': 0.1,       # 降低噪声强度，从 0.5 降到 0.1，保持扰动但避免数值爆炸
        'alpha': 0.01             # 恢复 alpha 为 0.01，先确保数值稳定再逐步加强物理约束
    }

    results = []
    folder = "./figures"
    os.makedirs(folder, exist_ok=True)

    print(f"\n--- Running Breakout Test with params: {best_params} ---")
    
    # 构建 args 对象
    args_dict = {
        'mode': best_params['mode'],
        'coef_negtive': best_params['coef_negtive'],
        'clamp_min_default': -6.0,
        'clamp_max_default': 6.0,
        'clamp_min_residual': -6.0,
        'clamp_max_residual': 6.0,
        'init_low': -1.0,
        'init_high': 1.0,
        'scheduler': False,
        'physical_function': best_params['physical_function'],
        'activation_gains': False,
        'softmax_center': False,
        'clamp_method': 'hard',
        'Nonlinear_method': 'shared',
        'clamp_min_learn': True
    }
    args = argparse.Namespace(**args_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    path = '/media/data/gaolili/deepLearning_project/pykan_physics_function_V2/data/conduct_function_fit_result'
    activation_config = None
    if os.path.exists(path):
        op = 'set'
        mode = args.mode
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
                    'init_low': -1.0,
                    'init_high': 1.0,
                }

    dim = 2
    np_i = 21
    np_b = 21
    ranges = [-1, 1]

    sol_fun = lambda x: torch.sin(torch.pi * x[:, [0]]) * torch.sin(torch.pi * x[:, [1]])
    source_fun = lambda x: -2 * torch.pi**2 * torch.sin(torch.pi * x[:, [0]]) * torch.sin(torch.pi * x[:, [1]])

    x_mesh = torch.linspace(ranges[0], ranges[1], steps=np_i)
    y_mesh = torch.linspace(ranges[0], ranges[1], steps=np_i)
    X, Y = torch.meshgrid(x_mesh, y_mesh, indexing="ij")
    x_i = torch.rand((np_i**2, 2)) * 2 - 1
    x_i = x_i.to(device)

    helper = lambda X, Y: torch.stack([X.reshape(-1,), Y.reshape(-1,)]).permute(1,0)
    xb1 = helper(X[0], Y[0])
    xb2 = helper(X[-1], Y[0])
    xb3 = helper(X[:,0], Y[:,0])
    xb4 = helper(X[:,0], Y[:,-1])
    x_b = torch.cat([xb1, xb2, xb3, xb4], dim=0)
    x_b = x_b.to(device)

    alpha = best_params['alpha']

    def batch_jacobian(func, x, create_graph=False):
        def _func_sum(x):
            return func(x).sum(dim=0)
        return autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1,0,2)

    def train_model(model_instance, current_epochs, current_lr, should_update_grid=False):
        optimizer = LBFGS(model_instance.parameters(), lr=current_lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)
        pbar = tqdm(range(current_epochs), desc='Training', ncols=100)

        for step in pbar:
            def closure():
                global pde_loss_val, bc_loss_val
                optimizer.zero_grad()
                
                sol_D1_fun = lambda x: batch_jacobian(model_instance, x, create_graph=True)[:,0,:]
                sol_D2 = batch_jacobian(sol_D1_fun, x_i, create_graph=True)
                lap = torch.sum(torch.diagonal(sol_D2, dim1=1, dim2=2), dim=1, keepdim=True)
                source = source_fun(x_i)
                pde_loss = torch.mean((lap - source)**2)
                pde_loss_val = pde_loss.item()

                bc_pred = model_instance(x_b)
                bc_true = sol_fun(x_b)
                bc_loss = torch.mean((bc_pred - bc_true)**2)
                bc_loss_val = bc_loss.item()

                loss = alpha * pde_loss + bc_loss
                loss.backward()
                return loss

            if should_update_grid and step % 5 == 0 and step < 50:
                model_instance.update_grid_from_samples(x_i)

            optimizer.step(closure)
            l2 = torch.mean((model_instance(x_i) - sol_fun(x_i))**2)

            pbar.set_description(f"pde_loss: {pde_loss_val:.2e} | bc_loss: {bc_loss_val:.2e} | l2: {l2.item():.2e}")
        return pde_loss_val, bc_loss_val, l2.item()

    # 使用所选参数初始化模型，并加入 noise_scale
    model = KAN(width=[dim, best_params['hidden_neurons'], 1], 
                grid=best_params['grid_size'], 
                k=3, 
                seed=1, 
                device=device, 
                activation_config=activation_config, 
                is_extend_grid=best_params['update_grid'],
                noise_scale=best_params['noise_scale'])

    print("Starting breakout initial training...")
    initial_pde_loss, initial_bc_loss, initial_l2 = train_model(model, best_params['epochs'], best_params['lr'], should_update_grid=best_params['update_grid'])

    # 暂时恢复符号回归以查看最终结果
    print("Applying symbolic logic...")
    model.fix_symbolic(0, 0, 0, 'x')
    model.fix_symbolic(0, 1, 0, 'x')
    model.fix_symbolic(0, 0, 1, 'x')
    model.fix_symbolic(0, 1, 1, 'x')
    # 对于 hn=10，我们需要为更多神经元设置符号，这里为了演示只设置一部分或保持样条
    for i in range(best_params['hidden_neurons']):
        model.fix_symbolic(1, i, 0, 'sin')

    print("Starting training after applying symbolic functions...")
    final_pde_loss, final_bc_loss, final_l2 = train_model(model, best_params['epochs'], best_params['lr'])

    print("Deriving symbolic formula...")
    formula = model.symbolic_formula()[0][0]
    final_formula_str = ex_round(formula, 6)
    print("Final Formula:", final_formula_str)

    print("\n--- Breakout Results Summary ---")
    print(f"Final PDE Loss: {final_pde_loss:.2e}")
    print(f"Final BC Loss: {final_bc_loss:.2e}")
    print(f"Final L2 Error: {final_l2:.2e}")
    print(f"Final Formula: {final_formula_str}")

if __name__ == "__main__":
    main()
