import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
import pandas as pd
import os

'''
基于XGBoost原生接口的回归
'''

# 读取文件原始数据
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
    y = df1["price"]
    # XGBoost训练过程
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    params = {
        'booster': 'gbtree', # 基于树的模型'gbtrree', 线性模型'gbliner'
        'objective': 'reg:gamma', #
        'gamma': 0,
        'max_depth': 8,
        'lambda': 1,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 3,
        'silent': 0,
        'eta': 0.3,
        'seed': 1000,
    }

    dtrain = xgb.DMatrix(X_train, y_train)
    num_rounds = 300
    plst = params.items()
    model = xgb.train(plst, dtrain, num_rounds)

    y_predict = model.predict(dtrain)
    v3 = list(y_predict)
    v4 = list(y_train)
    print(v4)  # 真实
    print(v3)  # 预测
    v0 = list(map(lambda x: 1 if x[1] * 0.8 <= x[0] <= x[1] * 1.2 else 0, zip(v4, v3)))
    accuracy_train = sum(v0) / len(v4)
    print(accuracy_train)

    # 对测试集进行预测
    dtest = xgb.DMatrix(X_test)
    ans = model.predict(dtest)
    v1 = list(ans)
    v2 = list(y_test)
    print(v2)  # 真实
    print(v1)  # 预测
    v = list(map(lambda x: 1 if x[1] * 0.8 <= x[0] <= x[1] * 1.2 else 0, zip(v2, v1)))
    accuracy_test = sum(v) / len(v2)
    print(accuracy_test)

    # 显示重要特征
    plot_importance(model)
    plt.show()