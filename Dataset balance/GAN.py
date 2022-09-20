import pandas as pd
import torch
import torch.nn as nn
import warnings
import numpy as np
warnings.filterwarnings('ignore')
path= "./SubMitoPred/"
dataset="M317Matrix_seqvec_end.txt"
idx_features_labels = np.genfromtxt("{}{}".format(path, dataset),
                               dtype=np.dtype(str))
X_train=idx_features_labels[0:317, 0:-1]
Y_train=idx_features_labels[0:317, -1]
y_train_smo=[]
y_test_smo=[]
for i in Y_train:
    y_train_smo.append(int(i))
process = pd.DataFrame(X_train,columns=[f'fea{i}' for i in range(1,X_train.shape[1] + 1)])
process['target'] = y_train_smo
X_for_generate = process.query("target == 1").iloc[:,:-1].values.astype(float)
X_non_default = process.query('target == 0').iloc[:,:-1].values.astype(float)
X_for_generate = torch.tensor(X_for_generate).type(torch.FloatTensor)

n_generate = X_non_default.shape[0] - X_for_generate.shape[0]
# # 超参数
BATCH_SIZE = 30
LR_G = 0.0001  # G生成器的学习率
LR_D = 0.0001  # D判别器的学习率
N_IDEAS = 20  # G生成器的初始想法(随机灵感)
# 搭建G生成器
G = nn.Sequential(  # 生成器
    nn.Linear(N_IDEAS, 128),  # 生成器等的随机想法
    nn.ReLU(),
    nn.Linear(128,768),
)

# 搭建D判别器
D = nn.Sequential(  # 判别器
    nn.Linear(768, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),  # 转换为0-1
)
# 定义判别器和生成器的优化器
opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

# GAN训练
for step in range(10):
    # 随机选取BATCH个真实的标签为1的样本.
    chosen_data = np.random.choice((X_for_generate.shape[0]), size=(BATCH_SIZE), replace=False)
    artist_paintings = X_for_generate[chosen_data, :]
    # 使用生成器生成虚假样本
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS, requires_grad=True)
    G_paintings = G(G_ideas)
    # 使用判别器得到判断的概率
    prob_artist1 = D(G_paintings)
    # 生成器损失
    G_loss = torch.mean(torch.log(1. - prob_artist1))
    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    prob_artist0 = D(artist_paintings)
    prob_artist1 = D(G_paintings.detach())
    # 判别器的损失
    D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)
    opt_D.step()
fake_data = G(torch.randn(n_generate,N_IDEAS)).detach().numpy()
X_default = pd.DataFrame(np.concatenate([X_for_generate,fake_data]),columns=[f'fea{i}' for i in range(1,X_train.shape[1] + 1)])
X_default['target'] = 1
X_non_default = pd.DataFrame(X_non_default,columns=[f'fea{i}' for i in range(1,X_train.shape[1] + 1)])
X_non_default['target'] = 0
train_data_gan = pd.concat([X_default,X_non_default])
X_gan = train_data_gan.iloc[:,:-1]
y_gan = train_data_gan.iloc[:,-1]
X_gan.to_csv('Vesicle Transport Proteins/Xm317MatrixSeq_GAN.txt',index=False,sep=' ',encoding='utf-8-sig')
y_gan.to_csv('Vesicle Transport Proteins/Ym317MatrixSeq_GAN.txt',index=False,sep=' ',encoding='utf-8-sig')
print(X_gan.shape,y_gan.shape)

