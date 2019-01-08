from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
import os
import pandas as pd
from sklearn.datasets import load_boston
path = "./"
list1 = os.listdir(path)
list1 = ["0.csv"]
for i in list1:
    print(i)
    Project_data = pd.read_csv(path + "/" + i, encoding="gbk")  # 读取预测数据

    df1 = Project_data[['Rent_amount', 'floor', 'Total_floor', 'Area', 'towards', 'Number_bedroom', 'Number_hall', 'Number_wei', 'location',
                        'subway_route', 'subway_site', 'distance', 'region','price']]
    m, n = df1.shape
    X = df1.iloc[:, 0: n - 1]
    Y = df1["price"]

    lr = LinearRegression()
    selector = RFECV(estimator=lr, cv=3)
    selector.fit(X, Y)
    print("N_features %s" % selector.n_features_)
    print("Support is %s" % selector.support_)
    print("Ranking %s" % selector.ranking_)
    print("Grid Scores %s" % selector.grid_scores_)
