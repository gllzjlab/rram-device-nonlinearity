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
        'lr': 0.001,               # Adam 用小学习率
        'hidden_neurons': 2,
        'grid_size': 5,
        'coef_negtive': False,
        'update_grid': False,
        'physical_function': 'set',
        'epochs': 100,
        'optimizer': 'Adam',       # Adam 替换 LBFGS
        'mode': 'real',
        'noise_scale': 0.001,
        'alpha': 0.01
    }

    results = []
    folder = "./figures"
    os.makedirs(folder, exist_ok=True)
    # 训练日志文件
    log_file = os.path.join(folder, "pde_training_C2_Adam.log")
    log_f = open(log_file, 'w')

    print(f"\n--- Running Breakout Test with params: {best_params} ---")
    log_f.write(f"Params: {best_params}\n")
    log_f.write("epoch,pde_loss,bc_loss,l2,phase\n")
    log_f.flush()
    
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

    def train_model(model_instance, current_epochs, current_lr, should_update_grid=False, phase_name="initial", optimizer_type="LBFGS"):
        if optimizer_type == "LBFGS":
            optimizer = LBFGS(model_instance.parameters(), lr=current_lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)
        else:
            optimizer = torch.optim.Adam(model_instance.parameters(), lr=current_lr)

        pbar = tqdm(range(current_epochs), desc='Training', ncols=100)
        loss_record = []
        loss_vals = [0.0, 0.0]   # [pde_loss_val, bc_loss_val]，可变容器避免闭包作用域问题

        for step in pbar:
            if optimizer_type == "LBFGS":
                def closure():
                    optimizer.zero_grad()
                    sol_D1_fun = lambda x: batch_jacobian(model_instance, x, create_graph=True)[:,0,:]
                    sol_D2 = batch_jacobian(sol_D1_fun, x_i, create_graph=True)
                    lap = torch.sum(torch.diagonal(sol_D2, dim1=1, dim2=2), dim=1, keepdim=True)
                    pde_loss = torch.mean((lap - source_fun(x_i))**2)
                    loss_vals[0] = pde_loss.item()
                    bc_loss = torch.mean((model_instance(x_b) - sol_fun(x_b))**2)
                    loss_vals[1] = bc_loss.item()
                    (alpha * pde_loss + bc_loss).backward()
                    return alpha * pde_loss + bc_loss

                if should_update_grid and step % 5 == 0 and step < 50:
                    model_instance.update_grid_from_samples(x_i)

                optimizer.step(closure)
            else:
                optimizer.zero_grad()
                sol_D1_fun = lambda x: batch_jacobian(model_instance, x, create_graph=True)[:,0,:]
                sol_D2 = batch_jacobian(sol_D1_fun, x_i, create_graph=True)
                lap = torch.sum(torch.diagonal(sol_D2, dim1=1, dim2=2), dim=1, keepdim=True)
                pde_loss = torch.mean((lap - source_fun(x_i))**2)
                loss_vals[0] = pde_loss.item()
                bc_loss = torch.mean((model_instance(x_b) - sol_fun(x_b))**2)
                loss_vals[1] = bc_loss.item()
                loss = alpha * pde_loss + bc_loss
                loss.backward()
                optimizer.step()

            pde_loss_val, bc_loss_val = loss_vals

            l2 = torch.mean((model_instance(x_i) - sol_fun(x_i))**2)
            l2_val = l2.item()

            # 记录每步 loss
            loss_record.append((step, pde_loss_val, bc_loss_val, l2_val))
            log_f.write(f"{step},{pde_loss_val:.6e},{bc_loss_val:.6e},{l2_val:.6e},{phase_name}\n")
            if step % 10 == 0:
                log_f.flush()

            pbar.set_description(f"pde_loss: {pde_loss_val:.2e} | bc_loss: {bc_loss_val:.2e} | l2: {l2_val:.2e}")
        
        # 保存该阶段 loss 曲线图
        steps_arr, pde_arr, bc_arr, l2_arr = zip(*loss_record)
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(steps_arr, pde_arr, label='PDE Loss')
        plt.plot(steps_arr, bc_arr, label='BC Loss')
        plt.yscale('log')
        plt.xlabel('Step'); plt.ylabel('Loss')
        plt.legend(); plt.title(f'{phase_name} — Loss')
        plt.subplot(1,2,2)
        plt.plot(steps_arr, l2_arr, label='L2 Error', color='green')
        plt.yscale('log')
        plt.xlabel('Step'); plt.ylabel('L2 Error')
        plt.legend(); plt.title(f'{phase_name} — L2 Error')
        plt.tight_layout()
        plt.savefig(os.path.join(folder, f'pde_loss_{phase_name}.png'), dpi=150)
        plt.close()
        
        return pde_loss_val, bc_loss_val, l2_val

    # 使用所选参数初始化模型，并加入 noise_scale
    model = KAN(width=[dim, best_params['hidden_neurons'], 1], 
                grid=best_params['grid_size'], 
                k=3, 
                seed=1, 
                device=device, 
                activation_config=activation_config, 
                is_extend_grid=best_params['update_grid'],
                noise_scale=best_params['noise_scale'],
                use_c2=True)

    print("Starting breakout initial training...")
    initial_pde_loss, initial_bc_loss, initial_l2 = train_model(model, best_params['epochs'], best_params['lr'], should_update_grid=best_params['update_grid'])

    # Phase1 激活函数可视化
    print("Plotting KAN activation functions (Phase1)...")
    model.plot(beta=10, folder=folder, save_name='PDE_activation_function_C2_Adam_phase1')
    print(f"Activation plot saved to: {folder}/")

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
    final_pde_loss, final_bc_loss, final_l2 = train_model(model, best_params['epochs'], best_params['lr'], phase_name="phase2_symbolic", optimizer_type=best_params['optimizer'])

    # Phase2 激活函数可视化
    print("Plotting KAN activation functions (Phase2)...")
    model.plot(beta=10, folder=folder, save_name='PDE_activation_function_C2_Adam_phase2')
    print(f"Activation plot saved to: {folder}/")

    print("Deriving symbolic formula...")
    formula = model.symbolic_formula()[0][0]
    final_formula_str = ex_round(formula, 6)
    print("Final Formula:", final_formula_str)

    print("\n--- Breakout Results Summary ---")
    print(f"Final PDE Loss: {final_pde_loss:.2e}")
    print(f"Final BC Loss: {final_bc_loss:.2e}")
    print(f"Final L2 Error: {final_l2:.2e}")
    print(f"Final Formula: {final_formula_str}")

    # 保存结果到文件
    result_file = os.path.join(folder, "pde_results_C2_Adam.txt")
    with open(result_file, 'w') as f:
        f.write(f"=== PDE Physical KAN (use_c2=True) Results ===\n")
        f.write(f"Params: {best_params}\n")
        f.write(f"Phase1 — Initial PDE Loss: {initial_pde_loss:.6e}\n")
        f.write(f"Phase1 — Initial BC Loss:  {initial_bc_loss:.6e}\n")
        f.write(f"Phase1 — Initial L2 Error: {initial_l2:.6e}\n")
        f.write(f"Phase2 — Final PDE Loss:   {final_pde_loss:.6e}\n")
        f.write(f"Phase2 — Final BC Loss:    {final_bc_loss:.6e}\n")
        f.write(f"Phase2 — Final L2 Error:   {final_l2:.6e}\n")
        f.write(f"Final Formula: {final_formula_str}\n")
    print(f"Results saved to: {result_file}")

    # 关闭日志文件
    log_f.write(f"# Final: pde_loss={final_pde_loss:.6e}, bc_loss={final_bc_loss:.6e}, l2={final_l2:.6e}, formula={final_formula_str}\n")
    log_f.close()
    print(f"Training log saved to: {log_file}")

if __name__ == "__main__":
    main()
