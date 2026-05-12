#!/usr/bin/env python
# coding: utf-8

# # Example 5: Special functions
# 
# Let's construct a dataset which contains special functions $f(x,y)={\rm exp}(J_0(20x)+y^2)$, where $J_0(x)$ is the Bessel function.
import sys
sys.path.append("/media/data/gaolili/deepLearning_project/pykan-master/")
from kan import *
#import matplotlib
#matplotlib.use('Agg')  # 使用非交互后端，保存图片用
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = KAN(width=[2,1,1], grid=3, k=3, seed=2, device=device)
f = lambda x: torch.exp(torch.special.bessel_j0(20*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2, device=device)

# train the model
model.fit(dataset, opt="LBFGS", steps=20)

# Plot trained KAN, the bessel function shows up in the bettom left
model.plot()
# 保存图片
#########网格精细化##########
model = model.refine(20)
model.fit(dataset, opt="LBFGS", steps=20)
model.plot()
# 保存图片

# suggest_symbolic does not return anything that matches with it, since Bessel function isn't included in the default SYMBOLIC_LIB. We want to add Bessel to i.
###拟合某个指定位置的激活函数
model.suggest_symbolic(0,0,0)

# SYMBOLIC_LIB.keys()
print(SYMBOLIC_LIB.keys())

# add bessel function J0 to the symbolic library. we should include a name and a pytorch implementation. c is the complexity assigned to J0.
add_symbolic('J0', torch.special.bessel_j0, c=1)

# After adding Bessel, we check suggest_symbolic again
# J0 fitting is not very good
model.suggest_symbolic(0,0,0)

# The fitting r2 is still not high, this is because the ground truth is J0(20x) which involves 20 which is too large. our default search is in (-10,10). so we need to set the search range bigger in order to include 20. now J0 appears at the top of the list
model.suggest_symbolic(0,0,0,a_range=(-40,40))
