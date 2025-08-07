import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import pandas as pd
import os
import pickle

data = pd.read_csv(os.path.join('3_GPR_data','FLAC_defor_to_Defor_train.csv')).to_numpy()
X = data[:, :3]
y = data[:, 3]

# 2. 定义GPR模型
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0]*3, length_scale_bounds=(1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-4, normalize_y=True)

# 3. 拟合模型
gpr.fit(X, y)
# 4. 保存模型到文件
with open(os.path.join('3_GPR_data',"gpr_model.pkl"), "wb") as f:
    pickle.dump(gpr, f)

y_mean, y_std = gpr.predict(X, return_std=True)

mse = mean_squared_error(y, y_mean)
print("MSE:", mse)
# 5. 可视化某一个变量 vs 输出，假设我们只画 x1 的关系
plt.figure(figsize=(8, 5))
plt.plot((0,100),(0,100),c='k')
plt.scatter(y, y_mean, label="Train data", color='blue')
#plt.scatter(X_test[:, 0], y_mean, label="Predicted mean", color='red')
plt.fill_between(y, y_mean - 2*y_std, y_mean + 2*y_std, alpha=0.2, color='gray', label="±2σ confidence")
plt.xlabel("y (变形)")
plt.ylabel("Predicted 沉降")
plt.title("GPR预测及置信区间 (仅展示x1关系)")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join('3_GPR_data','1.png'),dpi=800)
