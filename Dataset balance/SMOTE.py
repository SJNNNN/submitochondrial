import warnings
import numpy as np
warnings.filterwarnings('ignore')
path= "SM424-18/inner/"
dataset="inter_space_seqvec_end.txt"
idx_features_labels = np.genfromtxt("{}{}".format(path, dataset),dtype=np.dtype(str))
X_test=idx_features_labels[:,0:-1].tolist()
Y_test=idx_features_labels[:,-1].tolist()
print(len(X_test))
from collections import Counter
# 查看所生成的样本类别分布，0和1样本比例9比1，属于类别不平衡数据
print(Counter(Y_test))
# 使用imlbearn库中上采样方法中的SMOTE接口
from imblearn.over_sampling import SMOTE
# 定义SMOTE模型，random_state相当于随机数种子的作用
smo = SMOTE(random_state=42)
X_test_smo, Y_test_smo = smo.fit_resample(X_test,Y_test)
f = open("SM424-18/inner/inter_space_seqvec_SMOTE.txt", 'w', encoding="utf-8")
for i in range(len(X_test_smo)):
   f.writelines(str(X_test_smo[i]).replace('[','').replace(']','').replace(',','')+' '+str(Y_test_smo[i])+'\n')
f.close()
print(len(X_test_smo))

