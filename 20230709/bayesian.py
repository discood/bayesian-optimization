import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import warnings

warnings.simplefilter('ignore')

# 変数
c = 2
resolution = 64
X_range = np.linspace(0, 6.3, resolution).reshape(-1,1)

def function1(x):
    return x*np.sin(x)

def function2(x):
    return x * np.cos(x)

def function3(x):
    return (-x + 2*np.pi) * np.sin(x)

#初期化
def initialize_XY(data_num):
    rng = np.random.RandomState(1)
    training_indices = rng.choice(np.arange(X_range.size), size = data_num, replace=False)
    X_initial= X_range[training_indices]
    Y_initial = np.squeeze(function1(X_initial))
    return X_initial, Y_initial

def initialize_XY_2(data_num):
    rng = np.random.RandomState(1)
    training_indices = rng.choice(np.arange(X_range.size), size = data_num, replace=False)
    X_initial= X_range[training_indices]
    Y_initial = np.squeeze(0.5 * function2(X_initial) + 0.5 * function3(X_initial))
    return X_initial, Y_initial

def ucb(mean, std, c):
    return mean.ravel() + c*std

#UCBでの予測、平均と分散を返す
def gpr_prediction(X, Y):
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3)) + WhiteKernel()
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
    gp.fit(X, Y)
    print(gp.kernel_)
    mean, std = gp.predict(X_range,return_std=True)
    return mean,std

#提案された点を観測点として追加
def ucb_proposal(mean,std):
    max_index = np.argmax(ucb(mean,std,c))
    X_next = X_range[max_index]
    Y_next = function1(X_next)
    return X_next, Y_next

def ucb_proposal_2(mean,std):
    max_index = np.argmax(ucb(mean,std,c))
    X_next = X_range[max_index]
    Y_next = 0.5 * function2(X_next) + 0.5 * function3(X_next)
    return X_next, Y_next

def addition(X, Y ,X_next,Y_next):
    X = np.append(X, X_next)
    X = X.reshape(-1,1)
    Y = np.append(Y, Y_next)
    return X, Y