import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
# 一、读取数据
data = pd.read_csv('data.csv')

# 二、数据清洗
'''
1、处理标签数据
2、获取前31列
3、去除无用/异常属性：
    a、'num_outbound_cmds'和'is_host_login'字段方差为零，即取值唯一；
    b、'service'为服务器类型，与是否攻击无关，该属性去掉
    c、'shell'列为缺失值
'''
data['label'] = data['label'].apply(lambda x: x[:-1])  # 1、处理标签数据
data_new = data.iloc[:, :31]                           # 2、获取前31列
not_use = ['num_outbound_cmds', 'is_host_login', 'service', 'shell']
features = [i for i in data_new.columns if i not in not_use]
data_new = data.loc[:, features]
data_new['label'] = data['label']

# 三、将所有标签统一归为5类
DOS = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop']
Probin = ['ipsweep', 'nmap', 'portsweep', 'satan']
R2L = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster']
U2R = ['buffer_overflow', 'loadmodule', 'perl', 'rootkit']

index_DOS = data_new['label'].apply(lambda x: x in DOS)         # 找出对应子类样本的逻辑索引
index_Probin = data_new['label'].apply(lambda x: x in Probin)
index_R2L = data_new['label'].apply(lambda x: x in R2L)
index_U2R = data_new['label'].apply(lambda x: x in U2R)

data_new.loc[index_DOS, 'label'] = 'DOS'
data_new.loc[index_Probin, 'label'] = 'Probin'
data_new.loc[index_R2L, 'label'] = 'R2L'
data_new.loc[index_U2R, 'label'] = 'U2R'

# import numpy as np
# x = np.array([0.2, 0.3])
# index = x > 0.25
# x[index]

# 四、分出连续型和离散型变量
features_discrete = ['land', 'protocol_type', 'flag', 'land', 'su_attempted', 'is_guest_login']  # 离散属性
features_consecutive = [i for i in features if i not in features_discrete]    # 连续属性
data_consecutive = data_new.loc[:, features_consecutive]
y = data_new['label']

# 五、利用决策树判定属性的重要性

Dct = DecisionTreeClassifier(max_depth=5, random_state=10).fit(data_consecutive, y)
export_graphviz(Dct, out_file='Dct.dot', feature_names=data_consecutive.columns)
'若想将Dct.dot文件转成pdf，则需进入Dct.dot所在路径，然后在cmd中输入：dot -Tpdf Dct.dot -o output.pdf'
'决策树可视化方法参考：https://blog.csdn.net/just_youhg/article/details/83687911'

# 六、对离散变量进行哑变量处理
data_discrete = data_new.loc[:, features_discrete]
data_discrete_new = pd.get_dummies(data_discrete)
X = pd.concat([data_consecutive, data_discrete_new], axis=1)
# ['a', 'b', 'c']
#    a  b  c
# a: 1, 0, 0
# b: 0, 1, 0
# c: 0, 0, 1

# 七、建模及评价

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
model = DecisionTreeClassifier().fit(X_tr, y_tr)   # 模型训练
model.score(X_te, y_te)    # 模型的测试精度

pre = model.predict(X_te)
res = pd.DataFrame(y_te)
res['pre'] = pre
res.to_csv('res.csv', index=None)


