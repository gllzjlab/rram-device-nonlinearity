import torch
import torch.nn as nn
from .utils_physical import get_evaluate_extended_model_torch_params

# ========== 批量激活函数核心类 ==========




class UnifiedSmoothActivationWithCorrection(nn.Module):
    def __init__(self,
                 args, breakpoints_set, model_funcs_set, params_set, value_domain_set, corrections_set, bound_values_set,
                 breakpoints_reset, model_funcs_reset, params_reset, value_domain_reset, corrections_reset, bound_values_reset,
                 sigma=0.1, init_low=-3., init_high=3.):
        """
        统一融合 set 和 reset 的所有段，支持平滑修正项（offset_y + offset_dy * delta_V）

        corrections: list of tuple (offset_y, offset_dy)，长度等于段数
                   corrections[0] 用于第1段之后（即第1段不用），所以 corrections[0] 对应第1段的修正
                   注意：corrections 长度 = 段数 - 1
        """
        super().__init__()
        self.args = args
        self.sigma = sigma
        self.segments = []
        self.correction_enabled = []  # 标记该段是否需要加修正项（除第一段外都加）
        self.register_buffer('centers', None)
        if args.physical_function == 'set':
            self.K = len(model_funcs_set)
            # 构建 segments 列表
            self._add_segments(breakpoints_set, model_funcs_set, params_set, value_domain_set, corrections_set,
                               bound_values_set, prefix='set')
        if args.physical_function == 'reset':
            self.K = len(model_funcs_reset)
            self._add_segments(breakpoints_reset, model_funcs_reset, params_reset, value_domain_reset,
                               corrections_reset, bound_values_reset, prefix='reset')
        if args.physical_function == 'all':
            self.K = len(model_funcs_set) + len(model_funcs_reset)
            # 构建 segments 列表
            self._add_segments(breakpoints_set, model_funcs_set, params_set, value_domain_set, corrections_set,
                               bound_values_set, prefix='set')
            self._add_segments(breakpoints_reset, model_funcs_reset, params_reset, value_domain_reset,
                               corrections_reset, bound_values_reset, prefix='reset')
        if args.activation_gains:
            self.gains = nn.Parameter(torch.ones(self.K))
        self.map_function = torch.nn.Sigmoid()
        self.clamp_min = nn.Parameter(torch.tensor(float(init_low)))
        self.clamp_max = nn.Parameter(torch.tensor(float(init_high)))
        # self.clamp_min = args.clip_range[0]
        # self.clamp_max = args.clip_range[1]
        # 初始化为 1/K，然后 softmax 确保归一化
        init_weight = 1.0 / (self.K)
        self.coef_act = nn.Parameter(torch.full((self.K,), init_weight))
        if args.clamp_method == 'scale_bias':
            self.scale = nn.Parameter(torch.tensor(1.0))  # 初始化为 1
            self.bias = nn.Parameter(torch.tensor(0.0))  # 初始化为 0


        # 注册 centers
        if args.softmax_center:
            centers = []
            for seg in self.segments:
                c = seg['center']
                if c == float('inf'):
                    c = 1e6
                elif c == -float('inf'):
                    c = -1e6
                centers.append(float(c))
            self.register_buffer('centers', torch.tensor(centers))


    def _add_segments(self, breakpoints, model_funcs, params, value_domain, corrections, bound_values,prefix):
        """添加一组 segment（set 或 reset）"""
        boundaries = [value_domain[0]] + list(breakpoints) + [value_domain[1]]

        for i, (func, param) in enumerate(zip(model_funcs, params)):
            left = boundaries[i]
            right = boundaries[i + 1]
            center = (left + right) / 2.0
            if left == -float('inf'): center = right - 1.0
            if right == float('inf'): center = left + 1.0
            #if (prefix, i) not in [('reset', 2), ('set', 1)]:
            self.segments.append({
                'center': center,
                'left': left,
                'right': right,
                'func': func,
                'params': param,
                'prefix': prefix,
                'segment_idx': i,
                'corrections':corrections[i],
                'bound_min':min(bound_values[i],bound_values[i+1]),
                'bound_max': max(bound_values[i],bound_values[i+1]),
            })

            # 第一段（i == 0）不加修正，其余加
            if i == 0:
                self.correction_enabled.append(False)
            else:
                # corrections[i-1] 对应第 i 段的修正（i >= 1）
                self.correction_enabled.append(True)

        # 存储 corrections：索引从 0 开始对应第1段以后
        # 我们将 corrections 存为 list of dict，与 segment 一一对应（第一段 dummy）
        # self.corrections = [{'offset_y': 0.0, 'offset_dy': 0.0}]  # 第0段 dummy
        # for corr in corrections:
        #     offset_y, offset_dy = corr
        #     self.corrections.append({
        #         'offset_y': offset_y,
        #         'offset_dy': offset_dy
        #     })


    def forward(self, x, clamp_min=None, clamp_max=None):
        """
        V: 当前电压，shape (*)
        last_V: 上一个电压点（用于 delta_V），可选，默认为 V - 一个小量（静态近似）
               若未提供，可设为 V - 1e-3（模拟左侧极限）
        """
        # clamp_min = clamp_min if clamp_min is not None else self.args.clamp_min_default
        # clamp_max = clamp_max if clamp_max is not None else self.args.clamp_max_default

        # clamp_min = x.min().detach().item()
        # clamp_max = x.max().detach().item()
        #x = torch.clamp(x, clamp_min, clamp_max)
        if self.args.clamp_method == 'scale_bias':
            x = self.scale * x + self.bias
            clamp_min = self.clamp_min
            clamp_max = self.clamp_max
        if self.args.clamp_method == 'hard':

            clamp_min = torch.tensor(0.0, device=x.device) if not self.args.clamp_min_learn else self.clamp_min
            clamp_max = self.clamp_max

            # clamp_min = -3.
            # clamp_max = 3.


            x = torch.clamp(x, clamp_min, clamp_max)
        if self.args.clamp_method == 'software':
            x = self.map_function(x)
            clamp_min = 0.0
            clamp_max = 1.0
        shape = x.shape
        x_flat = x.flatten()  # [N]
        N = x_flat.size(0)
        x_org = x_flat


        # 扩展 V 和 last_V 为 [K, N]
        if self.args.softmax_center:
            x_exp = x_flat.unsqueeze(0).expand(self.K, -1)  # [K, N]
            centers_exp = self.centers.unsqueeze(1).expand(-1, x_flat.size(0))  # [K, N]
            #
            # # --- 计算权重：基于距离的 softmax ---
            distances = (x_exp - centers_exp) / self.sigma
            logits = -distances ** 2
            weights_exp = torch.softmax(logits, dim=0)  # [K, N]

        ###########softmax 正权重##################
        else:
            weights = torch.softmax(self.coef_act, dim=0)  # [K]
            weights_exp = weights.unsqueeze(1).expand(-1, N)


        # --- 计算每个 segment 的输出（带修正） ---
        outputs = []
        for k, seg in enumerate(self.segments):
            # seg_min_org = min(seg['left'], seg['right'])
            # seg_max_org = max(seg['left'], seg['right'])
            # if seg_max_org == 2.:
            #     seg_max = seg_max_org - 0.12 * (seg_max_org - seg_min_org)
            # else:
            #     seg_max = seg_max_org - 0.05 * (seg_max_org - seg_min_org)
            # ####real的5种导电机制用以下的seg_min和seg_max，效果较差
            # seg_min = seg_min_org + 0.05 * (seg_max_org - seg_min_org)


            ####real的5种导电机制用以下的seg_min和seg_max，效果较好
            if  seg['prefix'] == 'set':
                if seg['segment_idx'] == 0:
                    seg_min = min(seg['left'], seg['right'])
                else:
                    seg_min = min(seg['left'], seg['right']) + 5e-3
                seg_max = max(seg['left'], seg['right'])
            else:
                if seg['segment_idx'] == 0:
                    seg_max = max(seg['left'], seg['right'])
                else:
                    seg_max = max(seg['left'], seg['right']) - 0.0125
                seg_min = min(seg['left'], seg['right'])

            bound_min = seg['bound_min']
            bound_max = seg['bound_max']
            x_mapped = (x_org - clamp_min)/(clamp_max - clamp_min) * (seg_max - seg_min) + seg_min
            func = seg['func']
            param_tensors = [torch.tensor(p, dtype=x.dtype, device=x.device) for p in seg['params']]

            # 基础模型输出
            base_out = func(x_mapped, *param_tensors)  # [N]

            # 是否加修正项
            if self.correction_enabled[k]:
                corr = seg['corrections']# k >= 1 才有实际修正
                offset_y = torch.tensor(corr['offset_y'], dtype=x.dtype, device=x.device)
                offset_dy = torch.tensor(corr['offset_dy'], dtype=x.dtype, device=x.device)
                delta_x = x_mapped - seg['left']
                #correction_term = offset_y + offset_dy * delta_x
                final_out = base_out + offset_y + offset_dy * delta_x
            else:
                final_out = base_out  # 第0段不加修正
            # final_min = final_out.min()
            # final_max = final_out.max()
            if self.args.activation_gains:
                final_out = (final_out / bound_max) * self.gains[k]
            else:
                final_out /= bound_max
            #final_out = (final_out - bound_min) / (bound_max - bound_min)
            outputs.append(final_out)

        # 堆叠输出: [K, N]
        outputs_stack = torch.stack(outputs, dim=0)  # [K, N]

        # --- 加权求和 ---
        fused_output = torch.sum(weights_exp * outputs_stack, dim=0)  # [N]

        return fused_output.reshape(shape)
        #return outputs_stack.reshape((-1, shape[0], shape[1])).permute(1, 2, 0)



class UnifiedSmoothActivationWithCorrection_1(nn.Module):
    def __init__(self,
                 args, breakpoints_set, model_funcs_set, params_set, value_domain_set, corrections_set, bound_values_set,
                 breakpoints_reset, model_funcs_reset, params_reset, value_domain_reset, corrections_reset, bound_values_reset,
                 sigma=0.1, init_low=-3., init_high=3.):
        """
        统一融合 set 和 reset 的所有段，支持平滑修正项（offset_y + offset_dy * delta_V）

        corrections: list of tuple (offset_y, offset_dy)，长度等于段数
                   corrections[0] 用于第1段之后（即第1段不用），所以 corrections[0] 对应第1段的修正
                   注意：corrections 长度 = 段数 - 1
        """
        super().__init__()
        self.args=args
        self.sigma = sigma
        self.segments = []
        self.correction_enabled = []  # 标记该段是否需要加修正项（除第一段外都加）
        self.register_buffer('centers', None)

        if args.physical_function == 'set':
            self.K = len(model_funcs_set)
            # 构建 segments 列表
            self._add_segments(breakpoints_set, model_funcs_set, params_set, value_domain_set, corrections_set,
                               bound_values_set, prefix='set')
        if args.physical_function == 'reset':
            self.K = len(model_funcs_reset)
            self._add_segments(breakpoints_reset, model_funcs_reset, params_reset, value_domain_reset,
                               corrections_reset, bound_values_reset, prefix='reset')
        if args.physical_function == 'all':
            self.K = len(model_funcs_set) + len(model_funcs_reset)
            # 构建 segments 列表
            self._add_segments(breakpoints_set, model_funcs_set, params_set, value_domain_set, corrections_set,
                               bound_values_set, prefix='set')
            self._add_segments(breakpoints_reset, model_funcs_reset, params_reset, value_domain_reset,
                               corrections_reset, bound_values_reset, prefix='reset')
        # self.clamp_min = -6
        # self.clamp_max = 6
        # self.clamp_min = args.clip_range[0]
        # self.clamp_max = args.clip_range[1]
        self.map_function = torch.nn.Sigmoid()
        self.clamp_min = nn.Parameter(torch.tensor(float(init_low)))
        self.clamp_max = nn.Parameter(torch.tensor(float(init_high)))

        self.epoch = 0

        # 初始化为 1/K，然后 softmax 确保归一化
        init_coeffs = torch.randn(self.K)  # 标准正态分布，有正有负

        # 然后进行归一化，确保和为1
        # 这里使用投影方法
        init_coeffs = init_coeffs - (torch.sum(init_coeffs) - 1) / (self.K)

        self.coef = nn.Parameter(init_coeffs)


        # 注册 centers
        # centers = []
        # for seg in self.segments:
        #     c = seg['center']
        #     if c == float('inf'):
        #         c = 1e6
        #     elif c == -float('inf'):
        #         c = -1e6
        #     centers.append(float(c))
        # self.register_buffer('centers', torch.tensor(centers))



    def _add_segments(self, breakpoints, model_funcs, params, value_domain, corrections, bound_values,prefix):
        """添加一组 segment（set 或 reset）"""
        boundaries = [value_domain[0]] + list(breakpoints) + [value_domain[1]]

        for i, (func, param) in enumerate(zip(model_funcs, params)):
            left = boundaries[i]
            right = boundaries[i + 1]
            # center = (left + right) / 2.0
            if left == -float('inf'): center = right - 1.0
            if right == float('inf'): center = left + 1.0
            #if (prefix, i) not in [('reset', 2)]:
            self.segments.append({
                #'center': center,
                'left': left,
                'right': right,
                'func': func,
                'params': param,
                'prefix': prefix,
                'segment_idx': i,
                'corrections':corrections[i],
                'bound_min':min(bound_values[i],bound_values[i+1]),
                'bound_max': max(bound_values[i],bound_values[i+1]),
            })

            # 第一段（i == 0）不加修正，其余加
            if i == 0:
                self.correction_enabled.append(False)
            else:
                # corrections[i-1] 对应第 i 段的修正（i >= 1）
                self.correction_enabled.append(True)

        # 存储 corrections：索引从 0 开始对应第1段以后
        # 我们将 corrections 存为 list of dict，与 segment 一一对应（第一段 dummy）
        # self.corrections = [{'offset_y': 0.0, 'offset_dy': 0.0}]  # 第0段 dummy
        # for corr in corrections:
        #     offset_y, offset_dy = corr
        #     self.corrections.append({
        #         'offset_y': offset_y,
        #         'offset_dy': offset_dy
        #     })


    def forward(self, x, clamp_min=None, clamp_max=None):
        """
        V: 当前电压，shape (*)
        last_V: 上一个电压点（用于 delta_V），可选，默认为 V - 一个小量（静态近似）
               若未提供，可设为 V - 1e-3（模拟左侧极限）
        """
        # if (x < -4).sum().item() / x.numel() > 0.1 or (x > 4).sum().item() / x.numel() > 0.1:
        #    print(f"超出±6范围的比例: {(x < -4).sum().item() / x.numel() * 100:.2f}% - {(x > 4).sum().item() / x.numel() * 100:.2f}%")

        # clamp_min = self.clamp_min
        # clamp_max = self.clamp_max

        # clamp_min = clamp_min if clamp_min is not None else self.args.clamp_min_default
        # clamp_max = clamp_max if clamp_max is not None else self.args.clamp_max_default

        if self.args.clamp_method == 'hard':
            clamp_min = self.clamp_min
            clamp_max = self.clamp_max

            x = torch.clamp(x, clamp_min, clamp_max)
        if self.args.clamp_method == 'software':
            x = self.map_function(x)
            clamp_min = 0
            clamp_max = 1

        # if self.training:
        #     # 只在训练初期动态调整（如前10个epoch）
        #     if self.epoch < 9:
        #         with torch.no_grad():
        #             # 计算更保守的分位数
        #             lower_quantile = torch.quantile(x, 0.10)  # 10% 分位数
        #             upper_quantile = torch.quantile(x, 0.90)  # 90% 分位数
        #
        #             # 与固定范围取并集，确保覆盖所有分段函数
        #             clamp_min = min(-6, lower_quantile - 0.5)
        #             clamp_max = max(6, upper_quantile + 0.5)
        #
        #             # 更新运行平均值（用于后续固定范围）
        #             if not hasattr(self, 'running_clamp_min'):
        #                 self.register_buffer('running_clamp_min', torch.tensor(clamp_min))
        #                 self.register_buffer('running_clamp_max', torch.tensor(clamp_max))
        #             else:
        #                 # 使用指数移动平均平滑更新
        #                 alpha = 0.1
        #                 self.running_clamp_min = alpha * clamp_min + (1 - alpha) * self.running_clamp_min
        #                 self.running_clamp_max = alpha * clamp_max + (1 - alpha) * self.running_clamp_max
        #     else:
        #         # 训练后期使用运行平均值
        #         clamp_min = self.running_clamp_min
        #         clamp_max = self.running_clamp_max
        # else:
        #     # 测试阶段：使用训练后期确定的固定范围
        #     clamp_min = self.running_clamp_min if hasattr(self, 'running_clamp_min') else -6
        #     clamp_max = self.running_clamp_max if hasattr(self, 'running_clamp_max') else 6
            #print(f"BN后数据范围: min={x.min().item():.4f}, max={x.max().item():.4f}")
            #print(f"BN后数据均值: {x.mean().item():.4f}, 标准差: {x.std().item():.4f}")
            # if (x < -6).sum().item() / x.numel() >0 or (x > 6).sum().item() / x.numel() > 0:
            #     print(f"超出±6范围的比例: {(x < -6).sum().item() / x.numel() * 100:.2f}% - {(x > 6).sum().item() / x.numel() * 100:.2f}%")


        shape = x.shape
        x_flat = x.flatten()  # [N]
        N = x_flat.size(0)
        x_org = x_flat




        ###########softmax 正权重##################
        # weights = torch.softmax(self.segment_weights, dim=0)  # [K]
        # weights_exp = weights.unsqueeze(1).expand(-1, N)

        weights_exp = self.coef.unsqueeze(1).expand(-1, N)
        #print('weights', weights_exp)
        # --- 计算每个 segment 的输出（带修正） ---
        outputs = []
        for k, seg in enumerate(self.segments):
            if  seg['prefix'] == 'set':
                if seg['segment_idx'] == 0:
                    seg_min = min(seg['left'], seg['right'])
                else:
                    seg_min = min(seg['left'], seg['right']) + 5e-3
                seg_max = max(seg['left'], seg['right'])
            else:
                if seg['segment_idx'] == 0:
                    seg_max = max(seg['left'], seg['right'])
                else:
                    seg_max = max(seg['left'], seg['right']) - 0.0125
                seg_min = min(seg['left'], seg['right'])


            # seg_min_org = min(seg['left'], seg['right'])
            # seg_max_org = max(seg['left'], seg['right'])
            #
            # seg_max = seg_max_org - 0.12 * (seg_max_org - seg_min_org)
            #
            # ####real的5种导电机制用以下的seg_min和seg_max，效果较差
            # seg_min = seg_min_org + 0.12 * (seg_max_org - seg_min_org)

            bound_min = seg['bound_min']
            bound_max = seg['bound_max']
            x_mapped = (x_org - clamp_min)/(clamp_max - clamp_min) * (seg_max - seg_min) + seg_min
            func = seg['func']
            param_tensors = [torch.tensor(p, dtype=x.dtype, device=x.device) for p in seg['params']]

            # 基础模型输出
            base_out = func(x_mapped, *param_tensors)  # [N]

            # 是否加修正项
            if self.correction_enabled[k]:
                corr = seg['corrections']# k >= 1 才有实际修正
                offset_y = torch.tensor(corr['offset_y'], dtype=x.dtype, device=x.device)
                offset_dy = torch.tensor(corr['offset_dy'], dtype=x.dtype, device=x.device)
                delta_x = x_mapped - seg['left']
                #correction_term = offset_y + offset_dy * delta_x
                final_out = base_out + offset_y + offset_dy * delta_x
            else:
                final_out = base_out  # 第0段不加修正
            final_out /= bound_max
            #final_out = (final_out - bound_min) / (bound_max - bound_min)
            outputs.append(final_out)

        # 堆叠输出: [K, N]
        outputs_stack = torch.stack(outputs, dim=0)  # [K, N]

        # --- 加权求和 ---
        fused_output = torch.sum(weights_exp * outputs_stack, dim=0)  # [N]

        return fused_output.reshape(shape)


if __name__ == "__main__":
    op = 'set'
    mode = 'real'
    path = '/media/data/gaolili/deepLearning_project/pykan_physics_function_V2/data/conduct_function_fit_result'
    breakpoints_set, model_funcs_set, params_set, value_domain_set, corrections_set, bound_values_set, = get_evaluate_extended_model_torch_params(mode=mode, op=op, path=path)
    op = 'reset'
    breakpoints_reset, model_funcs_reset, params_reset, value_domain_reset, corrections_reset, bound_values_reset, = get_evaluate_extended_model_torch_params(mode = mode, op=op, path=path)

    model = UnifiedSmoothActivationWithCorrection(breakpoints_set, model_funcs_set, params_set, value_domain_set, corrections_set, bound_values_set,
                                                  breakpoints_reset, model_funcs_reset, params_reset, value_domain_reset, corrections_reset, bound_values_reset,)

    x = torch.linspace(-4,4, 100, requires_grad=True)
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()
    a = 0