import torch




# def B_batch_physical(x, grid, physical_basic):
#     '''
#     模仿原版 B_batch，但输出物理基函数与网格线性插值的组合
#
#     Args:
#     -----
#         x : (batch, in_dim)
#         grid : (in_dim, G + 1) -> 物理网格点
#         physical_phi_func : 一个函数，输入 (batch, in_dim)，输出 (batch, in_dim, 5)
#
#     Returns:
#     --------
#         组合基函数值 : (batch, in_dim, (G+1) * 5)
#     '''
#     batch_size, in_dim = x.shape
#     num_grid_points = grid.shape[1]
#     num_phys = 5  # 你的 5 个机制
#
#     # 1. 计算 5 个物理基函数输出 (batch, in_dim, 5)
#     #x_norm = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-5)
#     phi = physical_basic(x)
#
#     # 2. 计算网格线性插值基 (Hat functions / k=1 B-spline)
#     # 这一步手动实现 k=1 的样条逻辑，保证连续性
#     # 结果形状: (batch, in_dim, num_grid_points)
#     x_val = x.unsqueeze(2)  # (batch, in_dim, 1)
#     grid_val = grid.unsqueeze(0)  # (1, in_dim, G+1)
#
#     # 计算每个 x 相对于网格点的线性贡献
#     # 这是一个简化版的 B_batch(k=1)
#     # 我们找到 x 所在的区间 [t_i, t_{i+1}]
#     # 只有在该区间相邻的两个网格点基函数才非零
#
#     # 计算所有相邻网格点之间的间距
#     h = grid[:, 1:] - grid[:, :-1]  # (in_dim, G)
#     h = h.unsqueeze(0)  # (1, in_dim, G)
#
#     # 计算 hat 函数 (线性插值基)
#     # 对于每个网格点 i，其基函数在 [t_{i-1}, t_i] 上升，在 [t_i, t_{i+1}] 下降
#     # 这里我们使用一个高效的向量化实现：
#     dist_to_grid = torch.abs(x_val - grid_val)
#     # 只保留相邻区间的贡献 (线性插值)
#     # 简单的做法是利用原版的 B_batch(k=1) 逻辑，但这里我们直接生成
#     # 满足 sum(grid_bases) = 1 且连续
#
#     grid_bases = torch.zeros(batch_size, in_dim, num_grid_points, device=x.device)
#
#     # 找到索引
#     ##得到左侧端点的索引############
#     idx = torch.searchsorted(grid, x.T.contiguous()).T - 1
#     idx = torch.clamp(idx, 0, num_grid_points - 2)
#
#     grid_3d = grid.unsqueeze(0).expand(batch_size, -1, -1)
#
#     # 2. 把 idx 变成 [Batch, In_dim, 1]
#     idx_3d = idx.unsqueeze(-1)
#     # 计算线性比例
#     #t_low的含义是：输入样本 $x$ 落在网格区间 $[t_{low}, t_{high}]$ 时，该区间左侧端点的物理坐标值。
#     t_low = torch.gather(grid_3d, 2, idx_3d).squeeze(-1)
#     t_high = torch.gather(grid_3d, 2, (idx + 1).unsqueeze(-1)).squeeze(-1)
#     ratio = (x - t_low) / (t_high - t_low + 1e-6)
#
#     # 填充基函数 (只有 low 和 high 两个位置有值)
#     # 这一步保证了权重的“网格连续性”
#     grid_bases.scatter_(2, idx.unsqueeze(2), (1 - ratio).unsqueeze(2))
#     grid_bases.scatter_(2, (idx + 1).unsqueeze(2), ratio.unsqueeze(2))
#
#     # 3. 组合：(物理函数) ⊗ (网格基)
#     # phi: (B, I, 5), grid_bases: (B, I, G+1)
#     # 我们想要 (B, I, (G+1)*5)
#     # 使用 einsum 或 outer product 展开
#     combined = torch.einsum('bif, big -> bigf', phi, grid_bases)
#
#     # 展平最后两个维度，适配原版 KAN 的 coef 乘法
#     return combined.reshape(batch_size, in_dim, -1)
#
#
#
#
# def coef2curve(x_eval, grid, coef, k, physical_basic, device="cpu"):
#     '''
#     converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).
#
#     Args:
#     -----
#         x_eval : 2D torch.tensor
#             shape (batch, in_dim)
#         grid : 2D torch.tensor
#             shape (in_dim, G+2k). G: the number of grid intervals; k: spline order.
#         coef : 3D torch.tensor
#             shape (in_dim, out_dim, G+k)
#         k : int
#             the piecewise polynomial order of splines.
#         device : str
#             devicde
#
#     Returns:
#     --------
#         y_eval : 3D torch.tensor
#             shape (batch, in_dim, out_dim)
#
#     '''
#
#     #b_splines = B_batch(x_eval, grid, k=k)
#     b_splines = B_batch_physical(x_eval, grid, physical_basic)
#     y_eval = torch.einsum('ijk,jlk->ijl', b_splines, coef.to(b_splines.device))
#
#     return y_eval
#
#
# def curve2coef(x_eval, y_eval, grid, k, physical_basic):
#     '''
#     converting B-spline curves to B-spline coefficients using least squares.
#
#     Args:
#     -----
#         x_eval : 2D torch.tensor
#             shape (batch, in_dim)
#         y_eval : 3D torch.tensor
#             shape (batch, in_dim, out_dim)
#         grid : 2D torch.tensor
#             shape (in_dim, grid+2*k)
#         k : int
#             spline order
#         lamb : float
#             regularized least square lambda
#
#     Returns:
#     --------
#         coef : 3D torch.tensor
#             shape (in_dim, out_dim, G+k)
#     '''
#     #print('haha', x_eval.shape, y_eval.shape, grid.shape)
#     batch = x_eval.shape[0]
#     in_dim = x_eval.shape[1]
#     out_dim = y_eval.shape[2]
#     n_coef = grid.shape[1] * 5
#     #mat = B_batch(x_eval, grid, k)
#     mat = B_batch_physical(x_eval, grid, physical_basic)
#     mat = mat.permute(1,0,2)[:,None,:,:].expand(in_dim, out_dim, batch, n_coef)
#     #print('mat', mat.shape)mat[
#     y_eval = y_eval.permute(1,2,0).unsqueeze(dim=3)
#     #print('y_eval', y_eval.shape)
#     device = mat.device
#
#     #coef = torch.linalg.lstsq(mat, y_eval, driver='gelsy' if device == 'cpu' else 'gels').solution[:,:,:,0]
#     try:
#         coef = torch.linalg.lstsq(mat, y_eval).solution[:,:,:,0]
#     except:
#         print('lstsq failed')
#
#     # manual psuedo-inverse
#     '''lamb=1e-8
#     XtX = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0,1,3,2), mat)
#     Xty = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0,1,3,2), y_eval)
#     n1, n2, n = XtX.shape[0], XtX.shape[1], XtX.shape[2]
#     identity = torch.eye(n,n)[None, None, :, :].expand(n1, n2, n, n).to(device)
#     A = XtX + lamb * identity
#     B = Xty
#     coef = (A.pinverse() @ B)[:,:,:,0]'''
#
#     return coef
#
#
def extend_grid(grid, k_extend=0):
    '''
    extend grid
    '''
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

    for i in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)

    return grid

def B_batch_physical_shared(x, grid, physical_basic):
    '''
    结构化共享版：参数量从 (G+1)*5 降回 (G+1)

    Args:
        phys_weights: (in_dim, 5) 的张量，代表每个输入维度对应的 5 个物理机制的混合比例。
                      如果为 None，默认平均分配。
    '''
    batch_size, in_dim = x.shape
    num_grid_points = grid.shape[1]

    # 1. 计算物理基函数 (batch, in_dim, 5)
    phi = physical_basic(x)

    grid_bases = torch.zeros(batch_size, in_dim, num_grid_points, device=x.device)
    idx = torch.searchsorted(grid, x.T.contiguous()).T - 1
    idx = torch.clamp(idx, 0, num_grid_points - 2)

    grid_3d = grid.unsqueeze(0).expand(batch_size, -1, -1)
    idx_3d = idx.unsqueeze(-1)
    t_low = torch.gather(grid_3d, 2, idx_3d).squeeze(-1)
    t_high = torch.gather(grid_3d, 2, (idx + 1).unsqueeze(-1)).squeeze(-1)
    ratio = (x - t_low) / (t_high - t_low + 1e-6)

    grid_bases.scatter_(2, idx.unsqueeze(2), (1 - ratio).unsqueeze(2))
    grid_bases.scatter_(2, (idx + 1).unsqueeze(2), ratio.unsqueeze(2))

    # 4. 组合：(综合物理信号) * (网格基)
    # phi_shared: (B, I), grid_bases: (B, I, G+1)
    # 结果: (B, I, G+1)
    combined = grid_bases * phi.unsqueeze(-1)

    return combined  # 返回 (batch, in_dim, G+1)


def coef2curve(x_eval, grid, coef, k, physical_basic, device="cpu"):
    # 此时 b_splines 是 (batch, in_dim, G+1)
    # coef 是 (in_dim, out_dim, G+1)
    # phys_weights 需要作为参数传入，或者在 model 内部维护
    # 这里演示不带 phys_weights 的版本，若要学习，需从 model.act_fun 获取
    b_splines = B_batch_physical_shared(x_eval, grid, physical_basic)
    y_eval = torch.einsum('ijk,jlk->ijl', b_splines, coef.to(b_splines.device))
    return y_eval


def curve2coef(x_eval, y_eval, grid, k, physical_basic):
    batch = x_eval.shape[0]
    in_dim = x_eval.shape[1]
    out_dim = y_eval.shape[2]

    # 此时 n_coef 就是网格点数 G+1，不再乘 5
    n_coef = grid.shape[1]

    mat = B_batch_physical_shared(x_eval, grid, physical_basic)
    mat = mat.permute(1, 0, 2)[:, None, :, :].expand(in_dim, out_dim, batch, n_coef)

    y_eval = y_eval.permute(1, 2, 0).unsqueeze(dim=3)

    try:
        coef = torch.linalg.lstsq(mat, y_eval).solution[:, :, :, 0]
    except:
        print('lstsq failed')
        coef = torch.zeros(in_dim, out_dim, n_coef).to(mat.device)

    return coef


def B_batch_physical_shared_C2(x, grid, physical_basic):
    '''
    二阶光滑（C2）物理基函数：使用五次多项式（Quintic Hermite Spline）代替线性插值。
    保证在网格点处的一阶和二阶导数为 0，从而实现全局 C2 连续性。

    Args:
        x: (batch, in_dim)
        grid: (in_dim, G + 1)
        physical_basic: 物理激活函数类
    '''
    batch_size, in_dim = x.shape
    num_grid_points = grid.shape[1]

    # 1. 计算物理基函数 (batch, in_dim, 5)
    phi = physical_basic(x)

    grid_bases = torch.zeros(batch_size, in_dim, num_grid_points, device=x.device)
    
    # 找到 x 所在的网格索引
    idx = torch.searchsorted(grid, x.T.contiguous()).T - 1
    idx = torch.clamp(idx, 0, num_grid_points - 2)

    grid_3d = grid.unsqueeze(0).expand(batch_size, -1, -1)
    idx_3d = idx.unsqueeze(-1)
    
    t_low = torch.gather(grid_3d, 2, idx_3d).squeeze(-1)
    t_high = torch.gather(grid_3d, 2, (idx + 1).unsqueeze(-1)).squeeze(-1)
    
    # 计算线性比例 ratio (0 到 1 之间)
    ratio = (x - t_low) / (t_high - t_low + 1e-6)
    
    # 使用五次多项式映射：s(t) = 6t^5 - 15t^4 + 10t^3
    # 该映射满足：s(0)=0, s'(0)=0, s''(0)=0; s(1)=1, s'(1)=0, s''(1)=0
    s = 6 * ratio**5 - 15 * ratio**4 + 10 * ratio**3
    
    # 填充基函数 (只有 low 和 high 两个位置有值，但通过 s(t) 实现了平滑过渡)
    grid_bases.scatter_(2, idx.unsqueeze(2), (1 - s).unsqueeze(2))
    grid_bases.scatter_(2, (idx + 1).unsqueeze(2), s.unsqueeze(2))

    # 组合物理信号：(物理信号) * (平滑网格基)
    combined = grid_bases * phi.unsqueeze(-1)

    return combined


def coef2curve_C2(x_eval, grid, coef, k, physical_basic, device="cpu"):
    '''使用 C2 光滑基函数的曲线计算'''
    b_splines = B_batch_physical_shared_C2(x_eval, grid, physical_basic)
    y_eval = torch.einsum('ijk,jlk->ijl', b_splines, coef.to(b_splines.device))
    return y_eval


def curve2coef_C2(x_eval, y_eval, grid, k, physical_basic):
    '''使用 C2 光滑基函数的系数拟合'''
    batch = x_eval.shape[0]
    in_dim = x_eval.shape[1]
    out_dim = y_eval.shape[2]
    n_coef = grid.shape[1]

    mat = B_batch_physical_shared_C2(x_eval, grid, physical_basic)
    mat = mat.permute(1, 0, 2)[:, None, :, :].expand(in_dim, out_dim, batch, n_coef)

    y_eval = y_eval.permute(1, 2, 0).unsqueeze(dim=3)

    try:
        coef = torch.linalg.lstsq(mat, y_eval).solution[:, :, :, 0]
    except:
        print('lstsq failed in C2')
        coef = torch.zeros(in_dim, out_dim, n_coef).to(mat.device)

    # 保护：如果 lstsq 返回 NaN（矩阵病态时可能非异常返回），替换为零
    if torch.isnan(coef).any():
        coef = torch.zeros(in_dim, out_dim, n_coef).to(mat.device)

    return coef
