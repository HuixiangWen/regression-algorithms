from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE


path = "./"
list1 = os.listdir(path)
list1 = ["train1.csv"]
for i in list1:
    print(i)
    Project_data = pd.read_csv(path + "/" + i, encoding="gbk")  # 读取预测数据

    df1 = Project_data[['Rent_amount', 'floor', 'Total_floor', 'Area', 'towards', 'Number_bedroom', 'Number_hall', 'Number_wei', 'location',
                        'subway_route', 'subway_site', 'distance', 'region','price']]
    from sklearn.model_selection import train_test_split

    import seaborn as sns

    sns.set()
    m, n = df1.shape
    X = df1.iloc[:, 0: n - 1]
    Y = df1["price"]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    test = pd.concat([x_test, y_test], axis=1)
    # train = pd.concat([x_train, y_train], axis=1)
    #
    # train_data = train.as_matrix()  # 将表格转化为矩阵，方便进行矩阵运算
    # cols = train_data.shape[1]  # j矩阵列数
    # train_x = train_data[:, 0:cols - 1]  # 全体样本，去掉标签
    # train_y = train_data[:, cols - 1:cols]  # 标签
    #
    # test_data = test.as_matrix()  # 将表格转化为矩阵，方便进行矩阵运算
    # cols = test_data.shape[1]  # j矩阵列数
    # test_x = test_data[:, 0:cols - 1]  # 全体样本，去掉标签
    # test_y = test_data[:, cols - 1:cols]  # 标签

    models = [
        Pipeline([
        ('Poly', PolynomialFeatures(degree=5)),
        ('Linear', LinearRegression())
        ])
    ]
    for model in models:
        model.fit(x_train, y_train)
        y_predict = model.predict(x_train)
        y_hat = model.predict(x_test)
        # print(y_predict)
        # train_score = model.score(train_x, train_y)
        # test_score = model.score(test_x, test_y)

        v3 = list(y_predict)
        v4 = list(y_train)
        v0 = list(map(lambda x: 1 if x[1] * 0.8 <= x[0] <= x[1] * 1.2 else 0, zip(v4, v3)))
        accuracy_train = sum(v0) / len(v4)
        print(accuracy_train)

        v1 = list(y_hat)
        v2 = list(y_test)
        v = list(map(lambda x: 1 if x[1]*0.8 <= x[0] <= x[1]*1.2 else 0, zip(v2, v1)))
        accuracy_test = sum(v)/len(v2)
        print(accuracy_test)

