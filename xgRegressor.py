import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
import os
import pandas as pd

'''
基于Scikit-learn接口的回归
'''

# 读取文件原始数据
path = "./"
list1 = os.listdir(path)
# list1 = ["train1.csv"]
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

    model = xgb.XGBRegressor(max_depth=6, learning_rate=0.2, n_estimators=160, silent=True, objective='reg:gamma')
    grid = model.fit(X_train, y_train)
    y_predict =grid.predict(X_train)
    v3 = list(y_predict)
    v4 = list(y_train)
    v0 = list(map(lambda x: 1 if x[1] * 0.8 <= x[0] <= x[1] * 1.2 else 0, zip(v4, v3)))
    accuracy_train = sum(v0) / len(v4)
    print(accuracy_train)

    # 对测试集进行预测
    ans = model.predict(X_test)
    v1 = list(ans)
    v2 = list(y_test)
    print(v1)#预测
    print(v2)#真实
    v = list(map(lambda x: 1 if x[1] * 0.8 <= x[0] <= x[1] * 1.2 else 0, zip(v2, v1)))
    accuracy_test = sum(v) / len(v2)
    print(accuracy_test)

    # 显示重要特征
    plot_importance(model)
    plt.show()