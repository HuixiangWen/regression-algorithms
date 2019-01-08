from sklearn.externals import joblib
from sklearn import neighbors
import pandas as pd
import os

path = "./newdata"
list1 = os.listdir(path)
print(list1)
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
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    from sklearn.preprocessing import StandardScaler

    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)

    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=6, max_depth=20)

    rf.fit(x_train, y_train)

    rf.score(x_test, y_test)

    print(rf.score(x_train, y_train))

    print(rf.score(x_test, y_test))

    # from sklearn.externals import joblib
    #
    # joblib.dump(rf, "./" + i[:-4] + ".m")

    # clf = joblib.load("1.m")