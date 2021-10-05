import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns# seaborn作为matplotlib的补充及扩展
from sklearn.model_selection import train_test_split
#数据清洗
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
full = pd.concat([train, test], ignore_index=True)
#如果设置为true，则无视表的index，直接合并，合并后生成新的index
print(full.head())
print('合并后的数据集',full.shape)
print(full.describe())
print(full.info())# 查看每一列的数据类型，和数据总数
# 其中数据类型列：年龄（Age）、船舱号（Cabin）里面有缺失数据：
# 1）年龄（Age）里面数据总数是 1046 条，缺失了 1309-1046=263
# 2）船票价格（Fare）里面数据总数是 1308 条，缺失了 1 条数据。
# 字符串列：
# 1）登船港口（Embarked）里面数据总数是 1307，只缺失了 2 条数据，缺失比较少
# 2）船舱号（Cabin）里面数据总数是 295，缺失了 1309-295=1014，缺失比较大。
# 这为我们下一步数据清洗指明了方向，只有知道哪些数据缺失数据，我们才能有针对性的处理
print('处理前：',full.isnull().sum())
#Age，Cabin，Embarked，Fare这些变量存在缺失值（Survived是预测值）
#填补fare值，fare与pclass相关用这个类别票价的中位数来填补
print(full[full.Fare.isnull()])#查看此人属于哪个类别
full.Fare.fillna(full[full.Pclass==3]['Fare'].median(),inplace=True)
#inplace 默认值False。如果为 Ture, 在原地填满。注意：这将修改此对象上的任何其他视图
#Embarked：查看Embarked的各个类别的数量，用众数替代缺省值
print(full.Embarked.value_counts())
#发现是S，用S代替缺省值
full.Embarked.fillna('S',inplace=True)
#Cabin缺失数据较多，缺失值用U填充，表示未知
full.Cabin.fillna('u',inplace=True)
#age:先根据‘Name’提取‘Title’，再用‘Title’的中位数对‘Age‘进行插补
full['Title'] = full['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
full.Title.value_counts()
for i in full['Title']:
    full.Age.fillna(full[full.Title==i]['Age'].median(),inplace=True)
print('处理后：',full.isnull().sum())
#特征提取
#age
# 主要用于对数据从最大值到最小值进行等距划分,使用 pd.cut 将年龄平均分成5个区间,数组长度跟原来数组一样
# full['AgeCut']=pd.cut(full.Age,5)
# #按照数据出现频率百分比划分，比如要把数据分为四份，则四段分别是数据的0-25%，25%-50%，50%-75%，75%-100%,数组长度跟原来数组一样
# # 使用 pd.cut 将费用平均分成5个区间
# full['FareCut']=pd.qcut(full.Fare,5)#必须用这种方法创建新的属性
#绘制Fare与Survived柱状图
full[['Fare','Survived']].groupby(['Fare']).mean().plot.bar(figsize=(8,5))
#plt.show()

print(pd.cut(full.Age,5).value_counts().sort_index())
print(pd.qcut(full.Fare,5).value_counts().sort_index())
# 根据各个分段 重新给 Age赋值
full.loc[full.Age<=16.136,'Age']=1
full.loc[(full.Age>16.136)&(full.Age<=32.102),'Age']=2
full.loc[(full.Age>32.102)&(full.Age<=48.068),'Age']=3
full.loc[(full.Age>48.068)&(full.Age<=64.034),'Age']=4
full.loc[full.Age>64.034,'Age']=5
# 根据各个分段 重新给 Fare赋值
full.loc[full.Fare<=7.854,'Fare']=1
full.loc[(full.Fare>7.854)&(full.Fare<=10.5),'Fare']=2
full.loc[(full.Fare>10.5)&(full.Fare<=21.558),'Fare']=3
full.loc[(full.Fare>21.558)&(full.Fare<=41.579),'Fare']=4
full.loc[full.Fare>41.579,'Fare']=5
sex_mapdict={'male':1,'female':0}
full['Sex']=full.Sex.map(sex_mapdict)
print(full.Sex.head())

#Embarked one hot编码
# one-hot的基本思想：将离散型特征的每一种取值都看成一种状态，若你的这一特征中有N个不相同的取值，那么我们就可以将该特征抽象成N种不同的状态，
# one-hot编码保证了每一个取值只会使得一种状态处于“激活态”，也就是说这N种状态中只有一个状态位值为1，其他状态位都是0
# 那么为什么使用one-hot编码？
# 我们之所以不使用标签编码，因为标签编码的问题是它假定类别值越高，该类别更好。显然在实际应用和生活中，这肯定不是一个好的方案，是我们所不能接受的。
# 我们使用one hot编码器对类别进行“二进制化”操作，然后将其作为模型训练的特征，原因正在于此。当然，如果我们在设计网络的时候考虑到这点，对标签编码的类别值进行特别处理，那就没问题。
# 不过，在大多数情况下，使用one hot编码是一个更简单直接的方案。另外，如果原本的标签编码是有序的，那one hot编码就不合适了——会丢失顺序信息
# 举个例子：
# 如果把客舱等级分为1，2，3，我们要预测的是泰坦尼克号的生存率，你知道每个人的生存率跟客舱等级具体有什么样的关系吗？
# 也就是说我们并不知道客舱等级是1，2，3哪个好，我们没办法对其量化，1，2，3只是一个标签，并不能对其进行分类，但是我们可以通过one-hot编码来量化这个特征是0还是1.也就是说可以找到这个特征存在于不存在与生存和死亡之间的关系。
embarkedDf = pd.DataFrame()
embarkedDf = pd.get_dummies(full['Embarked'], prefix='Embarked' )
full=pd.concat([full,embarkedDf],axis=1)
full.drop('Embarked',axis=1,inplace=True)
#print(full.head())
#Pclass one hot 编码
pclassDf = pd.DataFrame()
#使用get_dummies进行one-hot编码，列名前缀是Pclass
pclassDf = pd.get_dummies( full['Pclass'] , prefix='Pclass' )
full = pd.concat([full,pclassDf],axis=1)
#删掉客舱等级（Pclass）这一列
full.drop('Pclass',axis=1,inplace=True)
#print(full.head())
#Name中提取的Title进行 one hot 编码
titleDf = pd.DataFrame()
titleDf = pd.get_dummies(full['Title'] , prefix='Title' )
full = pd.concat([full,titleDf],axis=1)
#删掉客舱等级（Pclass）这一列
full.drop('Title',axis=1,inplace=True)
#print(full.head())
#同代直系亲属数（Parch）和不同代直系亲属数（SibSp）的处理,新建一个特征向量来描述这几个特征向量与存活率之间的关系
familyDf = pd.DataFrame()
familyDf[ 'FamilySize' ] = full[ 'Parch' ] + full[ 'SibSp' ] + 1
familyDf[ 'Family_Single' ] = familyDf[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
familyDf[ 'Family_Small' ]  = familyDf[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
familyDf[ 'Family_Large' ]  = familyDf[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )
full = pd.concat([full,familyDf],axis=1)
full.drop('FamilySize',axis=1,inplace=True)
#查看相关系数
corrDf=full.corr()
print(corrDf['Survived'].sort_values(ascending=False))
full_X = pd.concat([titleDf,#头衔
                     pclassDf,#客舱等级
                     familyDf,#家庭大小
                     full['Fare'],#船票价格
                     full['Sex'],#性别
                     embarkedDf,#登船港口
                    ] , axis=1 )

#原始数据集：特征
sourceRow=891
source_X = full_X.loc[0:sourceRow-1,:]
#原始数据集：标签
source_y = full.loc[0:sourceRow-1,'Survived']
#预测数据集：特征
pred_X = full_X.loc[sourceRow:,:]
x_train,x_test,y_train,y_test=train_test_split(source_X,source_y,test_size=0.3,random_state=5)#训练集每次随机抽取25%，所以每次准确率不太一样
#random_state使每次运行出的准确率都是一样的
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
print(model.predict(pred_X))
