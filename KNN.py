from sklearn.externals import joblib
from sklearn import neighbors
import pandas as pd
import os

path = "./"
list1 = os.listdir(path)
print(list1)
list1 = ["train1.csv"]
for i in list1:
    print(i)
    Project_data = pd.read_csv(path + "/" + i)  # 读取预测数据

    df1 = Project_data[['大类描述', '中类描述', '小类描述', '标准物料', '交货地点']]
    from sklearn.model_selection import train_test_split

    import seaborn as sns

    sns.set()
    m, n = df1.shape
    X = df1.iloc[:, 0: n - 1]
    Y = df1["交货地点"]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    test = pd.concat([x_test, y_test], axis=1)

    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)

    clf = neighbors.KNeighborsClassifier(algorithm='kd_tree', leaf_size=30, n_neighbors=5)  # 训练KNN分类器
    clf.fit(x_train, y_train)

    clf.fit(x_train, y_train.ravel())

    print(clf.score(x_train, y_train))  # 精度
    y_hat = clf.predict(x_train)
    # show_accuracy(y_hat, y_train, '训练集')
    print(clf.score(x_test, y_test))
    y_hat = clf.predict(x_test)

    # show_accuracy(y_hat, y_test, '测试集')
    knn_y_score = clf.predict_proba(x_test)
    print(list(knn_y_score))
    # from sklearn.externals import joblib
    #
    # joblib.dump(clf, "./" + i[:-4] + ".m")