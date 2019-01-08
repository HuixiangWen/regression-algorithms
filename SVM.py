from sklearn.externals import joblib
from sklearn import neighbors
import pandas as pd
import os
from sklearn.svm import SVC
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

    clf = SVC(C=0.1, kernel='rbf', gamma=30, decision_function_shape='ovo')

    clf.fit(x_train, y_train.ravel())

    print(clf.score(x_train, y_train))  # 精度

    y_hat = clf.predict(x_train)
    # show_accuracy(y_hat, y_train, '训练集')
    # print(y_hat[:100])
    # print(list(y_train)[:100])
    print(clf.score(x_test, y_test))

    y1_hat = clf.predict(x_test)
    # show_accuracy(y_hat, y_test, '测试集')
    # print(y1_hat[:100])
    # print(list(y_test)[:100])
    # from sklearn.externals import joblib
    #
    # joblib.dump(clf, "./" + i[:-4] + ".m")