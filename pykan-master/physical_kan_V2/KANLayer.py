import torch
import torch.nn as nn
import numpy as np
from .spline import *
from .utils import sparse_mask
from .physical_action_multi import UnifiedSmoothActivationWithCorrection, UnifiedSmoothActivationWithCorrection_1

class KANLayer(nn.Module):
    """
    KANLayer class
    

    Attributes:
    -----------
        in_dim: int
            input dimension
        out_dim: int
            output dimension
        num: int
            the number of grid intervals
        k: int
            the piecewise polynomial order of splines
        noise_scale: float
            spline scale at initialization
        coef: 2D torch.tensor
            coefficients of B-spline bases
        scale_base_mu: float
            magnitude of the residual function b(x) is drawn from N(mu, sigma^2), mu = sigma_base_mu
        scale_base_sigma: float
            magnitude of the residual function b(x) is drawn from N(mu, sigma^2), mu = sigma_base_sigma
        scale_sp: float
            mangitude of the spline function spline(x)
        base_fun: fun
            residual function b(x)
        mask: 1D torch.float
            mask of spline functions. setting some element of the mask to zero means setting the corresponding activation to zero function.
        grid_eps: float in [0,1]
            a hyperparameter used in update_grid_from_samples. When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
            the id of activation functions that are locked
        device: str
            device
    """

    def __init__(self, in_dim=3, out_dim=2, num=5, k=3,
                 noise_scale=0.5, scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0,
                 base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1],
                 sp_trainable=True, sb_trainable=True, save_plot_data = True, device='cpu', sparse_init=False,
                 is_extend_grid = True, coef_negative =True, activation_config = None, use_c2 = False):
        ''''
        initialize a KANLayer
        
        Args:
        -----
            in_dim : int
                input dimension. Default: 2.
            out_dim : int
                output dimension. Default: 3.
            num : int
                the number of grid intervals = G. Default: 5.
            k : int
                the order of piecewise polynomial. Default: 3.
            noise_scale : float
                the scale of noise injected at initialization. Default: 0.1.
            scale_base_mu : float
                the scale of the residual function b(x) is intialized to be N(scale_base_mu, scale_base_sigma^2).
            scale_base_sigma : float
                the scale of the residual function b(x) is intialized to be N(scale_base_mu, scale_base_sigma^2).
            scale_sp : float
                the scale of the base function spline(x).
            base_fun : function
                residual function b(x). Default: torch.nn.SiLU()
            grid_eps : float
                When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
            grid_range : list/np.array of shape (2,)
                setting the range of grids. Default: [-1,1].
            sp_trainable : bool
                If true, scale_sp is trainable
            sb_trainable : bool
                If true, scale_base is trainable
            device : str
                device
            sparse_init : bool
                if sparse_init = True, sparse initialization is applied.
            
        Returns:
        --------
            self
            
        Example
        -------
        >>> from kan.KANLayer import *
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> (model.in_dim, model.out_dim)
        '''
        super(KANLayer, self).__init__()
        # size 
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k

        # 根据 use_c2 参数绑定对应的基函数版本
        self.use_c2 = use_c2
        if self.use_c2:
            self.coef2curve_func = coef2curve_C2
            self.curve2coef_func = curve2coef_C2
        else:
            self.coef2curve_func = coef2curve
            self.curve2coef_func = curve2coef

        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None,:].expand(self.in_dim, num+1)
        if is_extend_grid:
            grid = extend_grid(grid, k_extend=k)

        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        noises = (torch.rand(self.num+1, self.in_dim, self.out_dim) - 1/2) * noise_scale / num

        if activation_config:
            if coef_negative:

                self.activation = UnifiedSmoothActivationWithCorrection_1(

                    **activation_config
                )
            else:
                self.activation = UnifiedSmoothActivationWithCorrection(

                    **activation_config
                )

        if is_extend_grid:
            self.coef = torch.nn.Parameter(curve2coef(self.grid[:,k:-k].permute(1,0), noises, self.grid, k, self.activation))
        else:
            self.coef = torch.nn.Parameter(
                curve2coef(self.grid.permute(1, 0), noises, self.grid, k, self.activation))

        if sparse_init:
            self.mask = torch.nn.Parameter(sparse_mask(in_dim, out_dim)).requires_grad_(False)
        else:
            self.mask = torch.nn.Parameter(torch.ones(in_dim, out_dim)).requires_grad_(False)
        
        self.scale_base = torch.nn.Parameter(scale_base_mu * 1 / np.sqrt(in_dim) + \
                         scale_base_sigma * (torch.rand(in_dim, out_dim)*2-1) * 1/np.sqrt(in_dim)).requires_grad_(sb_trainable)
        self.scale_sp = torch.nn.Parameter(torch.ones(in_dim, out_dim) * scale_sp * 1 / np.sqrt(in_dim) * self.mask).requires_grad_(sp_trainable)  # make scale trainable
        self.base_fun = base_fun




        self.grid_eps = grid_eps
        
        self.to(device)
        
    def to(self, device):
        super(KANLayer, self).to(device)
        self.device = device    
        return self

    def forward(self, x):
        '''
        KANLayer forward given input x
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            y : 2D torch.float
                outputs, shape (number of samples, output dimension)
            preacts : 3D torch.float
                fan out x into activations, shape (number of sampels, output dimension, input dimension)
            postacts : 3D torch.float
                the outputs of activation functions with preacts as inputs
            postspline : 3D torch.float
                the outputs of spline functions with preacts as inputs
        
        Example
        -------
        >>> from kan.KANLayer import *
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> x = torch.normal(0,1,size=(100,3))
        >>> y, preacts, postacts, postspline = model(x)
        >>> y.shape, preacts.shape, postacts.shape, postspline.shape
        '''
        batch = x.shape[0]
        preacts = x[:,None,:].clone().expand(batch, self.out_dim, self.in_dim)
            
        base = self.base_fun(x) # (batch, in_dim)
        #y = coef2curve_C2(x_eval=x, grid=self.grid, coef=self.coef, k=self.k, physical_basic=self.activation)
        y = self.coef2curve_func(x_eval=x, grid=self.grid, coef=self.coef, k=self.k, physical_basic=self.activation)
        postspline = y.clone().permute(0,2,1)
            
        y = self.scale_base[None,:,:] * base[:,:,None] + self.scale_sp[None,:,:] * y
        y = self.mask[None,:,:] * y
        
        postacts = y.clone().permute(0,2,1)
            
        y = torch.sum(y, dim=1)
        return y, preacts, postacts, postspline

    ###########基于样本更新网格###########

    # 功能：这是
    # KAN
    # 的“绝活”——网格自适应。
    #
    # 逻辑：
    #
    # 对输入数据
    # x
    # 进行排序（获取分布密度）。
    #
    # 混合网格：计算“等距网格”（均匀分布）和“分位数网格”（数据密集处点多）。
    #
    # 使用
    # grid_eps
    # 在两者之间插值。
    #
    # 无损迁移：最关键的一步，更新网格后立即调用
    # curve2coef，确保在新网格下激活函数的形状与旧网格时完全一致。

    def update_grid_from_samples(self, x, mode='sample'):
        '''
        update grid from samples
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            None
        
        Example
        -------
        >>> model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(model.grid.data)
        >>> x = torch.linspace(-3,3,steps=100)[:,None]
        >>> model.update_grid_from_samples(x)
        >>> print(model.grid.data)
        '''
        
        batch = x.shape[0]
        #x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(1, 0)
        x_pos = torch.sort(x, dim=0)[0]
        y_eval = self.coef2curve_func(x_pos, self.grid, self.coef, self.k, self.activation)
        num_interval = self.grid.shape[1] - 1 - 2*self.k
        
        def get_grid(num_interval):
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1,0)
            margin = 0.00
            h = (grid_adaptive[:,[-1]] - grid_adaptive[:,[0]] + 2 * margin)/num_interval
            grid_uniform = grid_adaptive[:,[0]] - margin + h * torch.arange(num_interval+1,)[None, :].to(x.device)
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid
        
        
        grid = get_grid(num_interval)
        
        if mode == 'grid':
            sample_grid = get_grid(2*num_interval)
            x_pos = sample_grid.permute(1,0)
            y_eval = self.coef2curve_func(x_pos, self.grid, self.coef, self.k)
        
        self.grid.data = extend_grid(grid, k_extend=self.k)
        #print('x_pos 2', x_pos.shape)
        #print('y_eval 2', y_eval.shape)
        self.coef.data = self.curve2coef_func(x_pos, y_eval, self.grid, self.k, self.activation)

    ########

    # 功能：用于“网络精细化”（Refining）。当你想要把一个
    # 5
    # 个网格点的模型变成
    # 10
    # 个点以提高精度时使用。
    #
    # 逻辑：
    #
    # 接收一个
    # parent
    # 层（旧层）。
    #
    # 在父层的网格基础上进行插值，生成更密的新网格。
    #
    # 通过最小二乘法将父层的曲线形状“继承”到当前层。这允许
    # KAN
    # 从一个粗糙的模型平滑切换到一个精细的模型，而不需要重新开始训练。
    def initialize_grid_from_parent(self, parent, x, mode='sample'):
        '''
        update grid from a parent KANLayer & samples
        
        Args:
        -----
            parent : KANLayer
                a parent KANLayer (whose grid is usually coarser than the current model)
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            None
          
        Example
        -------
        >>> batch = 100
        >>> parent_model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(parent_model.grid.data)
        >>> model = KANLayer(in_dim=1, out_dim=1, num=10, k=3)
        >>> x = torch.normal(0,1,size=(batch, 1))
        >>> model.initialize_grid_from_parent(parent_model, x)
        >>> print(model.grid.data)
        '''
        
        batch = x.shape[0]
        
        # shrink grid
        x_pos = torch.sort(x, dim=0)[0]
        y_eval = parent.coef2curve_func(x_pos, parent.grid, parent.coef, parent.k, parent.activation)
        num_interval = self.grid.shape[1] - 1 - 2*self.k
        
        
        '''
        # based on samples
        def get_grid(num_interval):
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1,0)
            h = (grid_adaptive[:,[-1]] - grid_adaptive[:,[0]])/num_interval
            grid_uniform = grid_adaptive[:,[0]] + h * torch.arange(num_interval+1,)[None, :].to(x.device)
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid'''
        
        #print('p', parent.grid)
        # based on interpolating parent grid
        def get_grid(num_interval):
            x_pos = parent.grid[:,parent.k:-parent.k]
            #print('x_pos', x_pos)
            sp2 = KANLayer(in_dim=1, out_dim=self.in_dim,k=1,num=x_pos.shape[1]-1,scale_base_mu=0.0, scale_base_sigma=0.0).to(x.device)

            #print('sp2_grid', sp2.grid[:,sp2.k:-sp2.k].permute(1,0).expand(-1,self.in_dim))
            #print('sp2_coef_shape', sp2.coef.shape)
            sp2_coef = curve2coef(sp2.grid[:,sp2.k:-sp2.k].permute(1,0).expand(-1,self.in_dim), x_pos.permute(1,0).unsqueeze(dim=2), sp2.grid[:,:], k=1).permute(1,0,2)
            shp = sp2_coef.shape
            #sp2_coef = torch.cat([torch.zeros(shp[0], shp[1], 1), sp2_coef, torch.zeros(shp[0], shp[1], 1)], dim=2)
            #print('sp2_coef',sp2_coef)
            #print(sp2.coef.shape)
            sp2.coef.data = sp2_coef
            percentile = torch.linspace(-1,1,self.num+1).to(self.device)
            grid = sp2(percentile.unsqueeze(dim=1))[0].permute(1,0)
            #print('c', grid)
            return grid
        
        grid = get_grid(num_interval)
        
        if mode == 'grid':
            sample_grid = get_grid(2*num_interval)
            x_pos = sample_grid.permute(1,0)
            y_eval = parent.coef2curve_func(x_pos, parent.grid, parent.coef, parent.k)
        
        grid = extend_grid(grid, k_extend=self.k)
        self.grid.data = grid
        self.coef.data = self.curve2coef_func(x_pos, y_eval, self.grid, self.k)

    ############剪枝
    ##功能：在网络剪枝（Pruning）后，提取出依然活跃的神经元。

    # 逻辑：
    #
    # 传入想要保留的输入 ID (in_id) 和输出 ID (out_id)。
    #
    # 创建一个新的、更小的 KANLayer。
    #
    # 把对应位置的 grid, coef, scale 等参数拷贝过去。
    #
    # 这通常在通过正则化识别出“没用”的神经元后执行，以减小模型体积。
    def get_subset(self, in_id, out_id):
        '''
        get a smaller KANLayer from a larger KANLayer (used for pruning)
        
        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons
            
        Returns:
        --------
            spb : KANLayer
            
        Example
        -------
        >>> kanlayer_large = KANLayer(in_dim=10, out_dim=10, num=5, k=3)
        >>> kanlayer_small = kanlayer_large.get_subset([0,9],[1,2,3])
        >>> kanlayer_small.in_dim, kanlayer_small.out_dim
        (2, 3)
        '''
        spb = KANLayer(len(in_id), len(out_id), self.num, self.k, base_fun=self.base_fun)
        spb.grid.data = self.grid[in_id]
        spb.coef.data = self.coef[in_id][:,out_id]
        spb.scale_base.data = self.scale_base[in_id][:,out_id]
        spb.scale_sp.data = self.scale_sp[in_id][:,out_id]
        spb.mask.data = self.mask[in_id][:,out_id]

        spb.in_dim = len(in_id)
        spb.out_dim = len(out_id)
        return spb
    
    #########交换神经元位置############
    # 功能：交换层内两个神经元的索引位置。
    #
    # 逻辑：
    #
    # 通过
    # with torch.no_grad() 确保交换操作不记录梯度。
    #
    # 同时交换
    # grid, coef, scale, mask
    # 等所有相关参数。
    #
    # 目的：主要用于可视化优化。KAN
    # 有一个功能可以将重要的连接移到中心，让画出来的网络结构图更整齐、更具可解释性。
    def swap(self, i1, i2, mode='in'):
        '''
        swap the i1 neuron with the i2 neuron in input (if mode == 'in') or output (if mode == 'out') 
        
        Args:
        -----
            i1 : int
            i2 : int
            mode : str
                mode = 'in' or 'out'
            
        Returns:
        --------
            None
            
        Example
        -------
        >>> from kan.KANLayer import *
        >>> model = KANLayer(in_dim=2, out_dim=2, num=5, k=3)
        >>> print(model.coef)
        >>> model.swap(0,1,mode='in')
        >>> print(model.coef)
        '''
        with torch.no_grad():
            def swap_(data, i1, i2, mode='in'):
                if mode == 'in':
                    data[i1], data[i2] = data[i2].clone(), data[i1].clone()
                elif mode == 'out':
                    data[:,i1], data[:,i2] = data[:,i2].clone(), data[:,i1].clone()

            if mode == 'in':
                swap_(self.grid.data, i1, i2, mode='in')
            swap_(self.coef.data, i1, i2, mode=mode)
            swap_(self.scale_base.data, i1, i2, mode=mode)
            swap_(self.scale_sp.data, i1, i2, mode=mode)
            swap_(self.mask.data, i1, i2, mode=mode)

