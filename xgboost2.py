import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
import pandas as pd

'''
基于XGBoost原生接口的分类
'''

train = pd.read_csv('train_clean2.csv')
test = pd.read_csv('test_clean2.csv')

selected_features = ['GENDER', 'AGE', 'TENURE', 'CHANNEL', 'AUTOPAY', 'ARPB_3M', 'CALL_PARTY_CNT', 'AFTERNOON_MOU', 'AVG_CALL_LENGTH']
X_train = train[selected_features]
X_test = test[selected_features]
y_train = train['BROADBAND']
y_test = test['BROADBAND']

# booster [default=gbtree]
# 有两中模型可以选择gbtree和gblinear。gbtree使用基于树的模型进行提升计算，gblinear使用线性模型进行提升计算。缺省值为gbtree。


params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 3,
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

plst = params.items()

dtrain = xgb.DMatrix(X_train, y_train)
num_rounds = 500
model = xgb.train(plst, dtrain, num_rounds)
ans0 = model.predict(dtrain)
# 计算训练准确率
cnt3 = 0
cnt4 = 0
for i in range(len(y_train)):
    if ans0[i] == y_train[i]:
        cnt3 += 1
    else:
        cnt4 += 1
print("Accuracy: %.2f %% " % (100 * cnt3 / (cnt3 + cnt4)))

# 对测试集进行预测
dtest = xgb.DMatrix(X_test)
ans = model.predict(dtest)
# 计算测试准确率
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    if ans[i] == y_test[i]:
        cnt1 += 1
    else:
        cnt2 += 1

print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))

