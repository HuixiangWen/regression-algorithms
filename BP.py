from sklearn.externals import joblib
import pandas as pd
import os
from sklearn.utils import shuffle

path = "./"
list1 = os.listdir(path)
list1 = ["0.csv"]
for i in list1:
    print(i)
    Project_data = pd.read_csv(path + "/" + i, encoding="gbk")  # 读取预测数据

    df1 = Project_data[['Rent_amount', 'floor', 'Total_floor', 'Area', 'towards', 'Number_bedroom', 'Number_hall', 'Number_wei', 'location',
                        'subway_route', 'subway_site', 'distance', 'region']]
    df2 = df1[df1["region"] == 1]
    df3 = df1[df1["region"] == 0].sample(int(df2.shape[0]), random_state=1)  # df.sample()采样
    df = shuffle(pd.concat([df2, df3], axis=0))  # 样本拼接并打乱

    from sklearn.model_selection import train_test_split
    import seaborn as sns
    sns.set()
    m, n = df1.shape
    X = df1.iloc[:, 0: n - 1]
    Y = df1["region"]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # from sklearn.preprocessing import StandardScaler
    # ss = StandardScaler()
    # x_train = ss.fit_transform(x_train)
    # x_test = ss.transform(x_test)

    from sklearn.neural_network import MLPClassifier

    model = MLPClassifier( hidden_layer_sizes=(80,60,20 ), max_iter=200 )

    model.fit(x_train,y_train)
    print((y_train.sum()))

    model.score(x_test,y_test)
    print((y_test.sum()))

    print(model.score(x_train,y_train))
    print(model.score(x_test,y_test))

    from sklearn.externals import joblib
    joblib.dump(model, "asd1.m")

