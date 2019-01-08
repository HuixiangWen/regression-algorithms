from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
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
    # test.to_csv("./t.csv", sep=",", index=0)

    from sklearn.preprocessing import MinMaxScaler

    ss = MinMaxScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)

    regressor = DecisionTreeClassifier(random_state=0)
    parameters = {'max_depth': range(5, 15)}
    scoring_fnc = make_scorer(accuracy_score)
    kfold = KFold(n_splits=10)
    grid = GridSearchCV(regressor, parameters, scoring_fnc, cv=kfold)
    grid = grid.fit(x_train, y_train.ravel())
    reg = grid.best_estimator_
    print('train score: %f' % grid.best_score_)
    print('best parameters:')
    for key in parameters.keys():
        print('%s: %d' % (key, reg.get_params()[key]))
    print('test score: %f' % reg.score(x_test, y_test))

    cart_y_score = grid.predict_proba(x_test)
    # print(list(cart_y_score))

    from sklearn.externals import joblib

    joblib.dump(grid, "./" + i[:-4] + ".m")

    joblib.dump(ss, "minmax.a")


    # clf = DecisionTreeClassifier(criterion='entropy', max_depth=20, min_samples_leaf=2)
    # clf.fit(x_train, y_train)
    #
    # clf.fit(x_train, y_train.ravel())
    #
    # print(clf.score(x_train, y_train))  # 精度
    # y_hat = clf.predict(x_train)
    # show_accuracy(y_hat, y_train, '训练集')

    # print(clf.score(x_test, y_test))
    # y_hat = clf.predict(x_test)
    # print(list(y_test))
    # print(list(y_hat))