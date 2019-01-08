import pandas as pd

# 加载数据集
df = pd.read_csv("train1.csv", header=0)

# 绘制图像
from matplotlib import pyplot as plt

# 从原始数据分理出需要的数据
x = df['region']
y = df['price']
# 绘图
plt.plot(x, y, 'r')
plt.scatter(x, y)

# 训练集和测试集的划分
train_df = df[:int(len(df)*0.7)]
test_df = df[int(len(df)*0.7):]

# 定义训练和测试使用的自变量和因变量
train_x = train_df['region'].values
train_y = train_df['price'].values

test_x = test_df['region'].values
test_y = test_df['price'].values

# 建立线性回归模型
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_x.reshape(len(train_x),1), train_y.reshape(len(train_y),1))
results = model.predict(test_x.reshape(len(test_x),1)) # 线性回归模型在测试集上的预测结果

# 线性回归误差计算
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

print("线性回归平均绝对误差: ", mean_absolute_error(test_y, results.flatten()))
print("线性回归均方误差: ", mean_squared_error(test_y, results.flatten()))

# 二次多项式预测
from sklearn.preprocessing import PolynomialFeatures

# 二次多项式回归特征矩阵
poly_features_2 = PolynomialFeatures(degree=2, include_bias=False)
poly_train_x_2 = poly_features_2.fit_transform(train_x.reshape(len(train_x),1))
poly_test_x_2 = poly_features_2.fit_transform(test_x.reshape(len(test_x),1))

# 二次多项式回归模型训练与预测
model = LinearRegression()
model.fit(poly_train_x_2, train_y.reshape(len(train_x),1)) # 训练模型

results_2 = model.predict(poly_test_x_2) # 预测结果
results_2.flatten() # 打印扁平化后的预测结果

print("二次多项式回归平均绝对误差: ", mean_absolute_error(test_y, results_2.flatten()))
print("二次多项式均方根误差: ", mean_squared_error(test_y, results_2.flatten()))


# 更高次多项式回归预测
from sklearn.pipeline import make_pipeline

train_x = train_x.reshape(len(train_x),1)
test_x = test_x.reshape(len(test_x),1)
train_y = train_y.reshape(len(train_y),1)

for m in [3, 4, 5]:
    model = make_pipeline(PolynomialFeatures(m, include_bias=False), LinearRegression())
    model.fit(train_x, train_y)
    pre_y = model.predict(test_x)
    print("{} 次多项式回归平均绝对误差: ".format(m), mean_absolute_error(test_y, pre_y.flatten()))
    print("{} 次多项式均方根误差: ".format(m), mean_squared_error(test_y, pre_y.flatten()))
    print("---")