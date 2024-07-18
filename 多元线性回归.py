import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
 
# 生成带有噪声的线性数据集
np.random.seed(0)
X = np.random.rand(100, 1)  # 特征
y = 3 * X.squeeze() + np.random.normal(0, 0.3, 100)  # 标签
 
# 没有使用 L2 正则化的线性回归模型
linear_model = LinearRegression()
linear_model.fit(X, y)
 
# 使用 L2 正则化的 Ridge 回归模型
ridge_model = Ridge(alpha=1.0)  # 正则化参数 alpha
ridge_model.fit(X, y)
 
# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, linear_model.predict(X), color='red', linewidth=2, label='Linear Regression (No L2 Regularization)')
plt.plot(X, ridge_model.predict(X), color='green', linewidth=2, label='Ridge Regression (L2 Regularization)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Comparison of Linear Regression with and without L2 Regularization')
plt.legend()
plt.show()