import numpy as np
import torch
from sklearn.linear_model import LinearRegression
import sympy
from torch.utils.data import Dataset
import random
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from autograd import grad
import autograd.numpy as npa
from scipy.misc import derivative
import os
from scipy.misc import derivative
# sigmoid = sympy.Function('sigmoid')
# name: (torch implementation, sympy implementation)
# SYMBOLIC_LIB = {'x': (lambda x: x, lambda x: x),
#                  'x^2': (lambda x: x**2, lambda x: x**2),
#                  'x^3': (lambda x: x**3, lambda x: x**3),
#                  'x^4': (lambda x: x**4, lambda x: x**4),
#                  '1/x': (lambda x: 1/x, lambda x: 1/x),
#                  '1/x^2': (lambda x: 1/x**2, lambda x: 1/x**2),
#                  '1/x^3': (lambda x: 1/x**3, lambda x: 1/x**3),
#                  '1/x^4': (lambda x: 1/x**4, lambda x: 1/x**4),
#                  'sqrt': (lambda x: torch.sqrt(x), lambda x: sympy.sqrt(x)),
#                  '1/sqrt(x)': (lambda x: 1/torch.sqrt(x), lambda x: 1/sympy.sqrt(x)),
#                  'exp': (lambda x: torch.exp(x), lambda x: sympy.exp(x)),
#                  'log': (lambda x: torch.log(x), lambda x: sympy.log(x)),
#                  'abs': (lambda x: torch.abs(x), lambda x: sympy.Abs(x)),
#                  'sin': (lambda x: torch.sin(x), lambda x: sympy.sin(x)),
#                  'tan': (lambda x: torch.tan(x), lambda x: sympy.tan(x)),
#                  'tanh': (lambda x: torch.tanh(x), lambda x: sympy.tanh(x)),
#                  'sigmoid': (lambda x: torch.sigmoid(x), sympy.Function('sigmoid')),
#                  #'relu': (lambda x: torch.relu(x), relu),
#                  'sgn': (lambda x: torch.sign(x), lambda x: sympy.sign(x)),
#                  'arcsin': (lambda x: torch.arcsin(x), lambda x: sympy.arcsin(x)),
#                  'arctan': (lambda x: torch.arctan(x), lambda x: sympy.atan(x)),
#                  'arctanh': (lambda x: torch.arctanh(x), lambda x: sympy.atanh(x)),
#                  '0': (lambda x: x*0, lambda x: x*0),
#                  'gaussian': (lambda x: torch.exp(-x**2), lambda x: sympy.exp(-x**2)),
#                  'cosh': (lambda x: torch.cosh(x), lambda x: sympy.cosh(x)),
#                  #'logcosh': (lambda x: torch.log(torch.cosh(x)), lambda x: sympy.log(sympy.cosh(x))),
#                  #'cosh^2': (lambda x: torch.cosh(x)**2, lambda x: sympy.cosh(x)**2),
# }

###########################
SYMBOLIC_LIB = {'schottky/Pool-Frenkel': (lambda x: torch.exp(torch.sqrt(x)), lambda x: sympy.exp(sympy.sqrt(x))),
                'Fowler-Nordheim':(lambda x: x**2*torch.exp(-1/x), lambda x: x**2*sympy.exp(-1/x)),
                'Direct': (lambda x: torch.exp(-1/torch.abs(x)), lambda x: sympy.exp(-1/sympy.Abs(x))),
                'Thermionic-Field':(lambda x: x*torch.exp(x**2), lambda x: x*sympy.exp(x**2)),
                'Hopping':(lambda x: torch.exp(x), lambda x: sympy.exp(x)),
                'Ohmic':(lambda x: x, lambda x: x),
                'Space-Charge-Limited':(lambda x: x**2, lambda x: x**2),
                'Ionic':(lambda x: torch.exp(1/x), lambda x: sympy.exp(1/x))
                }

SYMBOLIC_physical_fit_LIB = {
                'real-set':(lambda x: 19.5801 * x + 124.3383  * x**2 - 43.1261 * x**3 + 2.9993,
                                   lambda x: 19.5801 * x + 124.3383  * x**2 - 43.1261 * x**3 + 2.9993
                                   ),#三阶多项式
                'real-reset':(lambda x: -107.3047 * x + 95.4037  * x**2 + 28.0972 * x**3 - 12.9379,
                                    lambda x: -107.3047 * x + 95.4037  * x**2 + 28.0972 * x**3 - 12.9379
                                    ),#三阶多项式
                # 'reco-set':(lambda x: 194.7441 * x - 276.0198  * x**2 + 186.6775 * x**3 -5.4985,
                #                    lambda x: 194.7441 * x - 276.0198  * x**2 + 186.6775 * x**3 -5.4985
                #                    ),#三阶多项式
                # 'reco-reset':(lambda x: -172.9740 * x - 22.5201  * x**2 - 7.0792 * x**3 - 3.8860,
                #                     lambda x: -172.9740 * x - 22.5201  * x**2 - 7.0792 * x**3 - 3.8860
                #                     ),#三阶多项式
                }

def model_fowler(V, a, b, c, d):
    safe_input = a * V + b
    safe_input = torch.where(
        safe_input < 1e-8,
        torch.tensor(1e-8, dtype=safe_input.dtype, device=safe_input.device),
        safe_input
    )
    exponent = -1.0 / safe_input
    return c * (V ** 2) * torch.exp(exponent) + d

def model_direct(V, a, b, c, d):
    safe_input = torch.abs(a * V + b)
    safe_input = torch.where(
        safe_input < 1e-8,
        torch.tensor(1e-8, dtype=safe_input.dtype, device=safe_input.device),
        safe_input
    )
    return c * torch.exp(-1.0 / safe_input) + d

def model_thermionic(V, a, b, c, d):
    return c * V * torch.exp(a * (V ** 2) + b) + d

def model_space(V, a, b):
    return a * (V ** 2) + b

def model_ohmic(V, a, b):
    return a * V + b

def model_ionic(V, a, b, c, d):
    z = a * V + b
    return c * torch.exp(z + torch.sign(z)) + d

def model_schottky(V, a, b, c, d):
    z = a *V + b
    return c * torch.exp(torch.sqrt(torch.abs(z))) + d


def model_fowler_np(V, a, b, c, d):
    safe_input = a * V + b
    safe_input = np.where(
        safe_input < 1e-8,
        np.array(1e-8),
        safe_input
    )
    exponent = -1.0 / safe_input
    return c * (V ** 2) * np.exp(exponent) + d

def model_direct_np(V, a, b, c, d):
    safe_input = np.abs(a * V + b)
    safe_input = np.where(
        safe_input < 1e-8,
        np.array(1e-8),
        safe_input
    )
    return c * np.exp(-1.0 / safe_input) + d

def model_direct_np_inverse(I, a, b, c, d):
    temp = np.log((I - d)/c)

    safe_input = np.abs(a * V + b)
    safe_input = np.where(
        safe_input < 1e-8,
        np.array(1e-8),
        safe_input
    )
    return c * np.exp(-1.0 / safe_input) + d

def model_thermionic_np(V, a, b, c, d):
    return c * V * np.exp(a * (V ** 2) + b) + d

def model_space_np(V, a, b):
    return a * (V ** 2) + b

def model_ohmic_np(V, a, b):
    return a * V + b

def model_ionic_np(V, a, b, c, d):
    z = a * V + b
    return c * np.exp(z + np.sign(z)) + d

# 全局函数字典（函数名到函数的映射）
MODEL_FUNCS = {
    "fowler": model_fowler,
    "direct": model_direct,
    "thermionic": model_thermionic,
    "space": model_space,
    "ohmic": model_ohmic,
    "ionic": model_ionic
}


###############
def universal_smooth_model_torch(V, breakpoints, masks, model_funcs, model_params, value_domain):
    """
    PyTorch version of universal_smooth_model with C1 continuity.

    Args:
        V: (N,) torch.Tensor
        breakpoints: list of floats, length n_breaks
        masks: list of n_breaks+1 boolean tensors, one per segment
        model_funcs: list of callables [f0, f1, ...]
        model_params: list of lists, e.g. [[a0,b0], [a1,b1], ...]

    Returns:
        y: (N,) torch.Tensor
        corrections: list of dicts (for debug)
    """
    device = V.device
    n_breaks = len(breakpoints)
    boundaries = torch.tensor([value_domain[0]] + list(breakpoints) + [value_domain[1]])
    if len(model_funcs) != n_breaks + 1:
        raise ValueError("model_funcs should have length = n_breaks + 1")
    if len(model_params) != n_breaks + 1:
        raise ValueError("model_params should have length = n_breaks + 1")
    if len(masks) != n_breaks + 1:
        raise ValueError("masks should have length = n_breaks + 1")

    y = torch.zeros_like(V)
    corrections = []
    boundaries_value = []
    # corrections.append({'offset_y': 0.0, 'offset_dy': 0.0, 'func_name': getattr(f0, '__name__', 'unknown')})
    #corrections.append(None)
    # -------------------------------
    # Segment 0: no correction
    # -------------------------------
    mask0 = masks[0]
    f0 = model_funcs[0]
    p0 = model_params[0]
    if mask0.any():

        y[mask0] = f0(V[mask0], *p0)
        corrections.append({'offset_y': 0.0, 'offset_dy': 0.0})
        boundaries_value.append(y[0].item())
    # Record right endpoint of segment 0
    if n_breaks > 0:
        Vt0 = torch.tensor(breakpoints[0], device=device, dtype=torch.float32)
        f0_at_Vt0 = f0(Vt0.unsqueeze(0), *p0).item()
        # Compute derivative at Vt0
        Vt0_tensor = Vt0.requires_grad_(True)
        f0_out = f0(Vt0_tensor.unsqueeze(0), *p0)
        dy_dV = torch.autograd.grad(f0_out, Vt0_tensor, grad_outputs=torch.ones_like(f0_out), retain_graph=True)[0]
        last_y = f0_at_Vt0



        last_dy = dy_dV.item()
    else:
        last_y = last_dy = 0.0


    # -------------------------------
    # Segments 1 to n_breaks
    # -------------------------------
    for i in range(1, n_breaks + 1):
        mask = masks[i]


        fi = model_funcs[i]
        pi = model_params[i]

        # Define local model with parameters
        def local_model(v):
            return fi(v, *pi)

        # Evaluate at connection point (last breakpoint)
        Vt = torch.tensor(breakpoints[i-1], device=device, dtype=torch.float32)  # same as last_Vt
        Vt = Vt.requires_grad_(True)
        y_raw_tensor = local_model(Vt.unsqueeze(0))
        dy_raw_tensor = torch.autograd.grad(y_raw_tensor, Vt, retain_graph=True)[0]

        y_raw = y_raw_tensor.item()
        dy_raw = dy_raw_tensor.item()

        # Compute correction
        offset_y = last_y - y_raw
        # print("last_y", last_y)
        # print("y_raw", y_raw)
        # print("y_raw_type", y_raw.dtype)
        # print("offset_y", offset_y)
        boundaries_value.append(last_y)
        offset_dy = last_dy - dy_raw

        # Apply correction: y = f(V) + offset_y + offset_dy * (V - Vt)
        V_i = V[mask]
        base_val = local_model(V_i)
        corrected_val = base_val + offset_y + offset_dy * (V_i - Vt.item())
        y[mask] = corrected_val

        corrections.append({
            'offset_y': float(offset_y),
            'offset_dy': float(offset_dy),
            #'connect_point': float(Vt.item()),
            #'func_name': getattr(fi, '__name__', 'unknown')
        })

        # Update last_y, last_dy for next segment (if not last)
        if i < n_breaks:
            next_Vt = torch.tensor(breakpoints[i], device=device, dtype=torch.float32)
            next_Vt = next_Vt.requires_grad_(True)
            y_next_raw = local_model(next_Vt.unsqueeze(0))
            dy_next_raw = torch.autograd.grad(y_next_raw, next_Vt, retain_graph=True)[0]
            y_at_next = y_next_raw.item() + offset_y + offset_dy * (next_Vt.item() - Vt.item())
            last_y = y_at_next
            last_dy = dy_next_raw.item() + offset_dy  # careful: dy_raw is at Vt, not next_Vt

    boundaries_value.append(y[-1].item())

    return y, corrections, boundaries_value

def universal_smooth_model_torch_dtype32(V, breakpoints, masks, model_funcs, model_params, value_domain):
    """
    PyTorch version of universal_smooth_model with C1 continuity.

    Args:
        V: (N,) torch.Tensor
        breakpoints: list of floats, length n_breaks
        masks: list of n_breaks+1 boolean tensors, one per segment
        model_funcs: list of callables [f0, f1, ...]
        model_params: list of lists, e.g. [[a0,b0], [a1,b1], ...]

    Returns:
        y: (N,) torch.Tensor
        corrections: list of dicts (for debug)
    """
    device = V.device
    n_breaks = len(breakpoints)
    boundaries = torch.tensor([value_domain[0]] + list(breakpoints) + [value_domain[1]])
    if len(model_funcs) != n_breaks + 1:
        raise ValueError("model_funcs should have length = n_breaks + 1")
    if len(model_params) != n_breaks + 1:
        raise ValueError("model_params should have length = n_breaks + 1")
    if len(masks) != n_breaks + 1:
        raise ValueError("masks should have length = n_breaks + 1")

    y = torch.zeros_like(V)
    corrections = []
    boundaries_value = []
    # corrections.append({'offset_y': 0.0, 'offset_dy': 0.0, 'func_name': getattr(f0, '__name__', 'unknown')})
    #corrections.append(None)
    # -------------------------------
    # Segment 0: no correction
    # -------------------------------
    mask0 = masks[0]
    f0 = model_funcs[0]
    p0 = model_params[0]
    if mask0.any():

        y[mask0] = f0(V[mask0], *p0)
        corrections.append({'offset_y': 0.0, 'offset_dy': 0.0})
        boundaries_value.append(y[0].item())
    # Record right endpoint of segment 0
    if n_breaks > 0:
        Vt0 = torch.tensor(breakpoints[0], device=device, dtype=torch.float32)
        f0_at_Vt0 = f0(Vt0.unsqueeze(0), *p0).item()
        # Compute derivative at Vt0
        Vt0_tensor = Vt0.requires_grad_(True)
        f0_out = f0(Vt0_tensor.unsqueeze(0), *p0)
        dy_dV = torch.autograd.grad(f0_out, Vt0_tensor, grad_outputs=torch.ones_like(f0_out), retain_graph=True)[0]
        last_y = f0_at_Vt0



        last_dy = dy_dV.item()
    else:
        last_y = last_dy = 0.0


    # -------------------------------
    # Segments 1 to n_breaks
    # -------------------------------
    for i in range(1, n_breaks + 1):
        mask = masks[i]


        fi = model_funcs[i]
        pi = model_params[i]

        # Define local model with parameters
        def local_model(v):
            return fi(v, *pi)

        # Evaluate at connection point (last breakpoint)
        Vt = torch.tensor(breakpoints[i-1], device=device, dtype=torch.float32)  # same as last_Vt
        Vt = Vt.requires_grad_(True)
        y_raw_tensor = local_model(Vt.unsqueeze(0))
        dy_raw_tensor = torch.autograd.grad(y_raw_tensor, Vt, retain_graph=True)[0]

        y_raw = y_raw_tensor.item()
        dy_raw = dy_raw_tensor.item()

        # Compute correction
        offset_y = np.float32(last_y - y_raw)
        boundaries_value.append(last_y)
        offset_dy = np.float32(last_dy - dy_raw)

        # Apply correction: y = f(V) + offset_y + offset_dy * (V - Vt)
        V_i = V[mask]
        base_val = local_model(V_i)
        corrected_val = base_val + offset_y + offset_dy * (V_i - Vt.item())
        y[mask] = corrected_val

        corrections.append({
            'offset_y': offset_y,
            'offset_dy': offset_dy,
            #'connect_point': float(Vt.item()),
            #'func_name': getattr(fi, '__name__', 'unknown')
        })

        # Update last_y, last_dy for next segment (if not last)
        if i < n_breaks:
            next_Vt = torch.tensor(breakpoints[i], device=device, dtype=torch.float32)
            next_Vt = next_Vt.requires_grad_(True)
            y_next_raw = local_model(next_Vt.unsqueeze(0))
            dy_next_raw = torch.autograd.grad(y_next_raw, next_Vt, retain_graph=True)[0]
            y_at_next = y_next_raw.item() + offset_y + offset_dy * (next_Vt.item() - Vt.item())
            last_y = y_at_next
            last_dy = dy_next_raw.item() + offset_dy  # careful: dy_raw is at Vt, not next_Vt

    boundaries_value.append(y[-1].item())

    return y, corrections, boundaries_value

####获取set-reset阶段的导电机制拟合函数以及曲线
import re
import ast
def get_breakpoints_parames(fit_result_txt):
    with open(fit_result_txt, 'r') as f:
        text = f.readlines()[-1].strip()
        vt_match = re.search(r'Vt\s*=\s*\[([^\]]+)\]', text)
        vt_values = []
        if vt_match:
            # 提取并分割所有数值
            vt_str = vt_match.group(1)
            vt_values = [float(x.strip()) for x in re.split(r',\s*', vt_str)]

        # 匹配完整的 params 数组
        match = re.search(r'params\s*=\s*(\[\[.*?\]\])', text, re.DOTALL)
        if not match:
            return None

        array_str = match.group(1)

        try:
            # 安全解析为 Python 列表
            param_values = ast.literal_eval(array_str)
        except (SyntaxError, ValueError):
            # 解析失败时使用正则提取所有数值
            numbers = re.findall(r'-?\d*\.?\d+(?:[eE][-+]?\d+)?', array_str)
            param_values =  [float(num) for num in numbers]
    return vt_values, param_values

def get_mask(x, breakpoints, n_beraks, op):
    if op == 'set':
        masks = [x <= breakpoints[0]] + \
                      [(x > breakpoints[i]) & (x <= breakpoints[i + 1])
                       for i in range(n_beraks - 1)] + \
                      [x > breakpoints[-1]]
    else:
        masks = [x >= breakpoints[0]] + \
                [(x < breakpoints[i]) & (x >= breakpoints[i + 1])
                 for i in range(n_beraks - 1)] + \
                [x < breakpoints[-1]]
    return masks

def get_model_params():
    model_params = {}
    path = '/media/data/gaolili/deepLearning_project/pykan_physics_function_V2/data/conduct_function_fit_result'
    set_fit_result_txt = os.path.join(path, 'real_set_fit_result.txt')
    reset_fit_result_txt = os.path.join(path, 'real_reset_fit_result.txt')
    breakpoints_set, params_set = get_breakpoints_parames(set_fit_result_txt)
    breakpoints_reset, params_reset = get_breakpoints_parames(reset_fit_result_txt)
    model_params['model_direct_1'] = params_reset[0]
    model_params['model_thermionic'] = params_reset[1]
    model_params['model_space'] = params_reset[2]
    model_params['model_fowler'] = params_set[0]
    model_params['model_direct_2'] = params_set[1]
    return model_params

def plot_results_multisegment(voltages, currents,currents_dtype32):
    """可视化拟合结果"""

    from matplotlib import pyplot as plt
    """可视化拟合结果"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 8))

    plt.plot(voltages, currents, 'r-', linewidth=2.5, label=f'原始I-V', zorder=3)
    #plt.plot(voltages, currents_dtype32, 'b-', linewidth=2.5, label=f'FP32 I-V', zorder=3)


    plt.xlabel('电压 (V)')
    plt.ylabel('电流 (μA)')
    plt.title('I-V 曲线分段拟合')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    a = 0


def get_evaluate_extended_model_torch_params(mode = 'real', op = 'set', path=None):

    if op == 'set':
        value_domain = [0, 2]
        stage_num = 401
        max_voltage = 2.
        fit_result_txt = os.path.join(path, mode + '_set_fit_result.txt')
        voltage_list = torch.linspace(0., max_voltage, stage_num)
        if mode == 'real':
            model_funcs = [model_fowler, model_direct]
        else:
            model_funcs = [model_ohmic, model_direct]

    else:
        value_domain = [3, 0]
        stage_num = 401
        max_voltage = 3###取最大负电压的绝对值
        fit_result_txt = os.path.join(path, mode + '_reset_fit_result.txt')
        voltage_list = torch.linspace(max_voltage, 0, stage_num)
        if mode == 'real':
            ###负电压绝对值从大到小拟合的，拟合的第一段是[1.5,3]
            model_funcs = [model_direct, model_thermionic, model_space]
        else:
            model_funcs = [model_ionic, model_thermionic]

    breakpoints, params = get_breakpoints_parames(fit_result_txt)
    masks = get_mask(voltage_list, breakpoints, len(breakpoints), op)

    # model_params_tensor = []
    # for seg_params in raw_params:
    #     param_tensors = [
    #         torch.tensor(p, dtype=torch.float32).requires_grad_(False)
    #         for p in seg_params
    #     ]
    #     model_params_tensor.append(param_tensors)

    results, corrections, bound_values = universal_smooth_model_torch(voltage_list, breakpoints, masks,
                                                         model_funcs, params, value_domain)

    # results_dtype32, corrections_dtype32, bound_values_dtype32 = universal_smooth_model_torch_dtype32(voltage_list, breakpoints, masks,
    #                                                                   model_funcs, params, value_domain)
    #
    # plot_results_multisegment(voltage_list, results, results_dtype32)

    return breakpoints, model_funcs, params, value_domain, corrections, bound_values








# def model_fowler(V, a, b, c, d):
#     y = c * V ** 2 * npa.exp(-1/(a * V + b)) + d
#     return y
# def model_direct(V, a, b, c, d):
#     y = c * npa.exp(-1 / (npa.abs(a * V + b))) + d
#     return y
# def model_thermionic(V, a, b, c, d):
#     y = c * V * npa.exp(a * V ** 2 + b) + d
#     return y
# def model_space(V, a, b):
#     y = a * V ** 2  + b
#     return y
# def model_ohmic(V, a, b):
#     y = a * V  + b
#     return y
#
#
#
# def model_ionic(V, a, b, c, d):
#     y = c * npa.exp((a *V + b)+(npa.sign(a * V + b))) + d
#     return y







# def universal_smooth_model(V, breakpoints, masks, model_funcs, model_params):
#     """
#     完全通用的多段平滑模型，每段可使用任意 model_func，自动保证连接点处C1连续
#
#     参数:
#     -----------
#     V : array_like
#         输入电压值
#     breakpoints : list of float
#         分段点 [Vt1, Vt2, ...] (必须按op参数有序排列)
#     model_funcs : list of callable
#         每段的模型函数 [func1, func2, ...] (比分段点多1)
#     model_params : list of list
#         每段的参数 [[a1,b1,...], [a2,b2,...], ...]
#     op : str
#         'set' (升序) 或 'reset' (降序)，决定分段方向
#
#     返回:
#     --------
#     y : ndarray
#         输出值
#     """
#     # 参数检查
#     n_breaks = len(breakpoints)
#     if len(model_funcs) != n_breaks + 1:
#         raise ValueError("model_funcs 数量应比分段点多1")
#     if len(model_params) != n_breaks + 1:
#         raise ValueError("model_params 数量应比分段点多1")
#
#
#
#
#     # 初始化结果数组
#     result = np.zeros_like(V)
#     corrections = []
#     # 处理第一段（无约束）
#     mask = masks[0]
#     current_func = model_funcs[0]
#     result[mask] = current_func(V[mask], *model_params[0])
#
#     # 记录上一段的右端点信息（用于平滑连接）
#     last_Vt = breakpoints[0]
#     last_y = current_func(last_Vt, *model_params[0])
#     last_dy = derivative(lambda x: current_func(x, *model_params[0]), last_Vt, dx=1e-6)
#
#     # 第0段无修正
#     corrections.append({'offset_y': 0.0, 'offset_dy': 0.0})
#
#     # 处理中间段（需要平滑连接）
#     for i in range(1, n_breaks + 1):
#         mask = masks[i]
#         current_func = model_funcs[i]
#         current_params = model_params[i]
#
#         def local_model(v):
#             return current_func(v, *current_params)
#
#         # 计算当前段在连接点处的原始值
#         y_raw = local_model(last_Vt)
#         #dy_raw = derivative(local_model, last_Vt, dx=1e-6)
#         local_model_deriv = grad(local_model)
#         dy_raw = local_model_deriv(np.array(last_Vt))
#
#         # 计算修正量使函数平滑
#         offset_y = last_y - y_raw
#         offset_dy = last_dy - dy_raw
#
#         # 应用平滑修正：y = f(V) + offset_y + offset_dy*(V-Vt)
#         V_current = V[mask]
#         result[mask] = local_model(V_current) + offset_y + offset_dy * (V_current - last_Vt)
#
#         # 保存修正参数
#         corrections.append({
#             'offset_y': float(offset_y),
#             'offset_dy': float(offset_dy),
#             'connect_point': float(last_Vt),  # 连接点位置
#             'func_name': current_func.__name__
#         })
#
#         # 更新右断点信息（如果是最后一段则不需要）
#         if i < n_breaks:
#             last_Vt = breakpoints[i]
#             try:
#                 last_y = local_model(last_Vt) + offset_y + offset_dy * (
#                             last_Vt - last_Vt)
#                 last_dy = dy_raw + offset_dy
#             except:
#                 pass
#
#     return result, corrections
#
# def evaluate_extended_model(V, breakpoints, model_funcs, model_params, op=None):
#     """
#     统一计算 [0,4] 上的值，关于 x=2 对称
#     """
#     if op == 'set':
#         sym_point = 2
#         start_point = 0
#         end_point = 2
#     if op == 'reset':
#         sym_point = -3
#         start_point = -3
#         end_point = 0
#     V = np.asarray(V)
#     result = np.zeros_like(V)
#
#     # [0,2] 部分
#     if op == 'reset':
#         mask_left = (V >= sym_point)
#     if op == 'set':
#         mask_left = (V <= sym_point)
#     if np.any(mask_left):
#         V_sub = V[mask_left]
#         # 构造 masks
#         n_seg = len(model_funcs)
#         bps = [start_point] + list(breakpoints) + [end_point]
#         masks = []
#         for i in range(n_seg):
#             l, h = bps[i], bps[i+1]
#             m = (V_sub >= l) & (V_sub < h) if i < n_seg-1 else (V_sub >= l) & (V_sub <= h)
#             masks.append(m)
#         y_sub, _ = universal_smooth_model(V_sub, breakpoints, masks, model_funcs, model_params)
#         result[mask_left] = y_sub
#
#     # [2,4] 部分
#     if op == 'set':
#         mask_right = (V > sym_point)
#     if op == 'reset':
#         mask_right = (V < sym_point)
#     if np.any(mask_right):
#         V_sub = V[mask_right]
#         if op == 'set':
#             V_sym = 4 - V_sub  # 映射回 [0,2]
#         if op == 'reset':
#             V_sym = -6 - V_sub
#         # 构造 masks for V_sym
#         n_seg = len(model_funcs)
#         bps = [start_point] + list(breakpoints) + [end_point]
#         masks_sym = []
#         for i in range(n_seg):
#             l, h = bps[i], bps[i+1]
#             m = (V_sym >= l) & (V_sym < h) if i < n_seg-1 else (V_sym >= l) & (V_sym <= h)
#             masks_sym.append(m)
#         y_sym, _ = universal_smooth_model(V_sym, breakpoints, masks_sym, model_funcs, model_params)
#         result[mask_right] = y_sym
#
#     return result, _, _
#
#
# def get_mask(x, breakpoints, n_beraks):
#     masks = [x <= breakpoints[0]] + \
#                   [(x > breakpoints[i]) & (x <= breakpoints[i + 1])
#                    for i in range(n_beraks - 1)] + \
#                   [x > breakpoints[-1]]
#     return masks
#
# import os
# from matplotlib import pyplot as plt
# def get_corrections():
#     mode = 'reco'
#     path = '/media/data/gaolili/deepLearning_project/pykan_physics_function_V2/data/conduct_function_fit_result'
#     set_fit_result_txt = os.path.join(path, mode+'_set_fit_result.txt')
#     reset_fit_result_txt = os.path.join(path, mode+'_reset_fit_result.txt')
#     breakpoints_set, params_set = get_breakpoints_parames(set_fit_result_txt)
#
#     breakpoints_reset, params_reset = get_breakpoints_parames(reset_fit_result_txt)
#     # breakpoints = []
#     # breakpoints.extend(breakpoints_reset)
#     # breakpoints.append(0)
#     # breakpoints.extend(breakpoints_set)
#     # params = []
#     # params.extend(params_reset)
#     # params.extend(params_set)
#     max_reset_voltage = -3
#     max_set_voltage = 2
#     stage_set_num = 401
#     stage_reset_num = 401
#     set_voltage_list_p = np.round(np.linspace(0, max_set_voltage, stage_set_num), decimals=3)
#     set_voltage_list_p_sym = np.round(np.linspace(0, 2*max_set_voltage, 2*stage_set_num), decimals=3)
#     reset_voltage_list_n = np.round(np.linspace(max_reset_voltage, 0, stage_reset_num), decimals=3)
#     reset_voltage_list_n_sym = np.round(np.linspace(2*max_reset_voltage, 0, 2*stage_reset_num), decimals=3)
#     # x = []
#     # x.extend(reset_voltage_list_n)
#     # x.extend(set_voltage_list_p)
#     # x = np.array(x)
#     masks_set = get_mask(set_voltage_list_p, breakpoints_set, len(breakpoints_set))
#     masks_reset = get_mask(reset_voltage_list_n, breakpoints_reset, len(breakpoints_reset))
#     if mode == 'real':
#         model_funcs_set = [model_fowler, model_direct]
#         model_funcs_reset = [model_direct, model_thermionic, model_space]
#     if mode == 'reco':
#         model_funcs_set = [model_ohmic, model_direct]
#         model_funcs_reset = [model_thermionic, model_ionic]
#
#     result_set, corrections_set = universal_smooth_model(set_voltage_list_p, breakpoints_set, masks_set, model_funcs_set, params_set)
#     result_set_norm = (result_set - result_set.min()) / (
#             result_set.max() - result_set.min())
#     ####将set曲线关于x=2对称
#     result_set_sym, _, _ = evaluate_extended_model(set_voltage_list_p_sym, breakpoints_set, model_funcs_set, params_set, op='set')
#     result_set_sym_norm = (result_set_sym - result_set_sym.min())/(result_set_sym.max() - result_set_sym.min())
#
#
#     result_reset, corrections_reset = universal_smooth_model(reset_voltage_list_n, breakpoints_reset, masks_reset, model_funcs_reset, params_reset)
#
#     #####将reset曲线关于x=-3对称
#     result_reset_sym, _, _ = evaluate_extended_model(reset_voltage_list_n_sym, breakpoints_reset, model_funcs_reset,
#                                                                                      params_reset, op='reset')
#     result_reset_sym_norm = (result_reset_sym - result_reset_sym.min()) / (result_reset_sym.max() - result_reset_sym.min())
#
#
#     ###将reset曲线从[-3, 0]平移到[2, 5]的范围
#     shift = 5
#     result_reset, corrections_reset = universal_smooth_model(reset_voltage_list_n, breakpoints_reset, masks_reset,
#                                                              model_funcs_reset, params_reset)
#     result_reset_norm = (result_reset - result_reset.min()) / (
#                 result_reset.max() - result_reset.min())
#     reset_voltage_shift = reset_voltage_list_n + shift
#     voltage_all = np.concatenate([set_voltage_list_p, reset_voltage_shift])
#     result_all = np.concatenate([result_set_norm, result_reset_norm])
#
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
#     plt.rcParams['axes.unicode_minus'] = False
#     plt.figure(figsize=(12, 8))
#     #
#     #
#     #
#     plt.plot(voltage_all, result_all, 'r-', color='green', linewidth=2.5, label='set', zorder=3)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     # plt.title('构造函数')
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.7)
#
#     plt.savefig('/media/data/gaolili/deepLearning_project/pykan_physics_function_V2/data/reco_set_reset.png')
#     plt.show()
#
#
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
#     plt.rcParams['axes.unicode_minus'] = False
#     plt.figure(figsize=(12, 8))
#     #
#     #
#     #
#     plt.plot(set_voltage_list_p_sym, result_set_sym_norm, 'r-', color = 'blue', linewidth=2.5, label='set', zorder=3)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     #plt.title('构造函数')
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.7)
#
#     plt.savefig('/media/data/gaolili/deepLearning_project/pykan_physics_function_V2/data/reco_set_sym.png')
#     plt.show()
#     plt.close()
#
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
#     plt.rcParams['axes.unicode_minus'] = False
#     plt.figure(figsize=(12, 8))
#     #
#     #
#     #
#     plt.plot(reset_voltage_list_n_sym, result_reset_sym_norm, 'r-', color='red', linewidth=2.5, label='reset', zorder=3)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     #plt.title('构造函数')
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.savefig('/media/data/gaolili/deepLearning_project/pykan_physics_function_V2/data/reco_reset_sym.png')
#     return corrections_set, corrections_reset

    # x = np.hstack([reset_voltage_list_n[0:-1], set_voltage_list_p])
    # y_0 = (result_reset[-1] + result_set[0])/ 2
    # y = np.hstack([result_reset[0:-1], np.array([y_0]), result_set[1:]])
    #
    # plt.figure(figsize=(12, 8))
#
#
#
# plt.plot(x, y, 'r-', linewidth=2.5, label='set', zorder=3)
# plt.xlabel('电压 (V)')
# plt.ylabel('电流 (μA)')
# plt.title('I-V 曲线分段拟合')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
#
#
# plt.show()

def universal_smooth_model_numpy(V, breakpoints, masks, model_funcs, model_params, value_domain):
    """
    NumPy version of universal_smooth_model with C1 continuity.
    Replaces torch.autograd with numerical differentiation.

    Args:
        V: (N,) numpy.ndarray
        breakpoints: list of floats, length n_breaks
        masks: list of n_breaks+1 boolean arrays, one per segment
        model_funcs: list of callables [f0, f1, ...] — must accept numpy array input
        model_params: list of lists, e.g. [[a0,b0], [a1,b1], ...]
        value_domain: [start, end] for boundary

    Returns:
        y: (N,) numpy.ndarray
        corrections: list of dicts {'offset_y': float, 'offset_dy': float}
        boundaries_value: list of boundary y-values (including start & end)
    """
    n_breaks = len(breakpoints)
    if len(model_funcs) != n_breaks + 1:
        raise ValueError("model_funcs should have length = n_breaks + 1")
    if len(model_params) != n_breaks + 1:
        raise ValueError("model_params should have length = n_breaks + 1")
    if len(masks) != n_breaks + 1:
        raise ValueError("masks should have length = n_breaks + 1")

    y = np.zeros_like(V, dtype=np.float64)
    corrections = []
    boundaries_value = []

    # -------------------------------
    # Segment 0: no correction
    # -------------------------------
    mask0 = masks[0]
    f0 = model_funcs[0]
    p0 = model_params[0]

    if np.any(mask0):
        y[mask0] = f0(V[mask0], *p0)
        corrections.append({'offset_y': 0.0, 'offset_dy': 0.0})
        boundaries_value.append(float(y[0]))  # Start value

    # Record right endpoint of segment 0 (for C1 continuity with next segment)
    if n_breaks > 0:
        Vt0 = breakpoints[0]
        f0_at_Vt0 = f0(np.array([Vt0]), *p0)[0]  # Evaluate at breakpoint
        dy_dV = derivative(lambda x: f0(x, *p0), Vt0, dx=1e-6)# ← NumPy numerical derivative
        last_y = f0_at_Vt0
        last_dy = dy_dV
    else:
        last_y = last_dy = 0.0

    # -------------------------------
    # Segments 1 to n_breaks
    # -------------------------------
    for i in range(1, n_breaks + 1):
        mask = masks[i]
        fi = model_funcs[i]
        pi = model_params[i]


        # Evaluate model at connection point (left endpoint = previous breakpoint)
        Vt = breakpoints[i-1]
        y_raw = fi(np.array([Vt]), *pi)[0]
        dy_raw = derivative(lambda x: fi(x, *pi), Vt, dx=1e-6)
        # Compute correction to ensure C1 continuity
        offset_y = last_y - y_raw
        offset_dy = last_dy - dy_raw

        boundaries_value.append(last_y)

        # Apply correction: y = f(V) + offset_y + offset_dy * (V - Vt)
        if np.any(mask):
            V_i = V[mask]
            base_val = fi(V_i, *pi)
            corrected_val = base_val + offset_y + offset_dy * (V_i - Vt)
            y[mask] = corrected_val

        corrections.append({
            'offset_y': float(offset_y),
            'offset_dy': float(offset_dy),
        })

        # Update last_y, last_dy for next segment (if not last)
        if i < n_breaks:
            next_Vt = breakpoints[i]
            y_next_raw = fi(np.array([next_Vt]), *pi)[0]
            dy_next_raw = derivative(lambda x: fi(x, *pi), next_Vt, dx=1e-6)

            # Compute corrected value & derivative at next breakpoint
            y_at_next = y_next_raw + offset_y + offset_dy * (next_Vt - Vt)
            dy_at_next = dy_next_raw + offset_dy  # Derivative correction is constant
            last_y = y_at_next
            last_dy = dy_at_next

    boundaries_value.append(float(y[-1]))  # End value

    return y, corrections, boundaries_value


def get_evaluate_extended_model_numpy_params(mode='real', op='set', path=None):
    """
    改为纯 NumPy 实现，无 PyTorch 依赖
    """

    if op == 'set':
        value_domain = [0, 2]
        stage_num = 401
        max_voltage = 2.0
        fit_result_txt = os.path.join(path, mode + '_set_fit_result.txt')
        voltage_list = np.linspace(0., max_voltage, stage_num)  # ← torch → numpy
        if mode == 'real':
            model_funcs = [model_fowler_np, model_direct_np]
        else:
            model_funcs = [model_ohmic_np, model_direct_np]

    else:  # op == 'reset'
        value_domain = [3, 0]
        stage_num = 401
        max_voltage = 3.0  # 取最大负电压绝对值
        fit_result_txt = os.path.join(path, mode + '_reset_fit_result.txt')
        voltage_list = np.linspace(max_voltage, 0, stage_num)  # ← torch → numpy
        if mode == 'real':
            # 负电压绝对值从大到小拟合，第一段是 [1.5, 3]
            model_funcs = [model_direct_np, model_thermionic_np, model_space_np]
        else:
            model_funcs = [model_ionic_np, model_thermionic_np]

    # 假设这些函数已适配 NumPy 输入
    breakpoints, params = get_breakpoints_parames(fit_result_txt)
    masks = get_mask(voltage_list, breakpoints, len(breakpoints), op)

    # 核心计算函数：需确保是 NumPy 实现
    results, corrections, bound_values = universal_smooth_model_numpy(
        voltage_list, breakpoints, masks, model_funcs, params, value_domain
    )

    return breakpoints, model_funcs, params, value_domain, corrections, bound_values


if __name__ == "__main__":
    x = torch.linspace(3, 0, 401, requires_grad=True)
    op = 'set'
    mode = 'reco'
    path = '/media/data/gaolili/deepLearning_project/pykan_physics_function_V2/data/conduct_function_fit_result'
    breakpoints_set, model_funcs_set, params_set, value_domain_set, corrections_set, bound_values_set = \
         get_evaluate_extended_model_torch_params(mode=mode, op=op, path=path)
    op = 'reset'
    breakpoints_reset, model_funcs_reset, params_reset, value_domain_reset, corrections_reset, bound_values_reset = \
        get_evaluate_extended_model_torch_params(mode = mode, op=op, path=path)

    a = 0