# 导入相关模块
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import random
import os
import pandas as pd
import sys
'''注：回归树的叶节点的数据类型是连续型，节点的值是‘一系列’数的均值'''
# 导入数据
# sys.exit()

path = "./"
list1 = os.listdir(path)
list1 = ["train1.csv"]
for i in list1:
    print(i)
    Project_data = pd.read_csv(path + "/" + i, encoding="gbk")  # 读取预测数据

    df1 = Project_data[['Rent_amount', 'floor', 'Total_floor', 'Area', 'towards', 'Number_bedroom', 'Number_hall', 'Number_wei', 'location',
                        'subway_route', 'subway_site', 'distance', 'region', 'price']]
    from sklearn.model_selection import train_test_split

    import seaborn as sns

    sns.set()
    m, n = df1.shape
    X = df1.iloc[:, 0: n - 1]
    Y = df1["price"]
    # 随机选取25%的数据构建测试样本，剩余作为训练样本
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    test = pd.concat([x_test, y_test], axis=1)
    # 初始化回归树模型
    depth = [18,20]
    p = 0
    for j in depth:
        p = p + 1
        decision_j = DecisionTreeRegressor(criterion='mse',max_depth=j)
        # 训练回归树模型
        grid = decision_j.fit(x_train,y_train)
        # 使用测试数据检验回归树结
        y_pre_decision_j = decision_j.predict(x_test)
        decision_score = decision_j.score(x_test, y_test)
        # print("预测值为：")
        # print(y_pre_decision_j)
        # print("真实值为：")
        # print(list(y_test))
        # print(decision_score)
        y_predict = grid.predict(x_train)
        v3 = list(y_predict)
        v4 = list(y_train)
        v0 = list(map(lambda x: 1 if x[1] * 0.8 <= x[0] <= x[1] * 1.2 else 0, zip(v4, v3)))
        accuracy_train = sum(v0) / len(v4)
        print(accuracy_train)
        v1 = list(y_pre_decision_j)
        v2 = list(y_test)
        v = list(map(lambda x: 1 if x[1]*0.8 <= x[0] <= x[1]*1.2 else 0, zip(v2, v1)))
        accuracy_test = sum(v)/len(v2)
        print(accuracy_test)

        # from sklearn.externals import joblib
        #
        # joblib.dump(grid, "./" + i[:-4] + ".m")

