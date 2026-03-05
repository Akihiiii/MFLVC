import torch
from network import Network
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# MNIST-USPS
# BDGP
# CCV
# Fashion
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
# ALOI-100
# Mfeat
# UCI
# COIL20
Dataname = 'UCI'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0001)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=1500)
parser.add_argument("--con_epochs", default=100)
parser.add_argument("--tune_epochs", default=150)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The code has been optimized.
# The seed was fixed for the performance reproduction, which was higher than the values shown in the paper.
if args.dataset == "MNIST-USPS":
    args.con_epochs = 50
    seed = 10
if args.dataset == "BDGP":
    args.con_epochs = 10
    seed = 10
if args.dataset == "CCV":
    args.con_epochs = 50
    seed = 3
if args.dataset == "Fashion":
    args.con_epochs = 100
    seed = 10
if args.dataset == "Caltech-2V":
    args.con_epochs = 50
    seed = 10
if args.dataset == "Caltech-3V":
    args.con_epochs = 50
    seed = 10
if args.dataset == "Caltech-4V":
    args.con_epochs = 50
    seed = 10
if args.dataset == "Caltech-5V":
    args.con_epochs = 50
    seed = 5
if args.dataset == "COIL20":
    args.con_epochs = 50
    seed = 5

if args.dataset == "Mfeat":
    args.con_epochs = 50
    seed = 10

if args.dataset == "ALOI-100":
    args.con_epochs = 50
    seed = 10

if args.dataset == "UCI":
    args.con_epochs = 50
    seed = 10

# Per-dataset batch size: small datasets need smaller batch to avoid excessive drop_last waste
if args.dataset in ["COIL20", "Mfeat", "UCI"]:
    args.batch_size = 256
if args.dataset == "ALOI-100":
    args.batch_size = 1024


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


# setup_seed(seed)


dataset, dims, view, data_size, class_num = load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

import torch.nn.functional as F

@torch.no_grad()
def sinkhorn_knopp(scores, epsilon=0.05, iterations=3):
    """
    计算最优传输矩阵 (Sinkhorn 算法)
    :param scores: 特征与原型的内积相似度矩阵, shape: [batch_size, class_num]
    :param epsilon: 熵正则化系数，控制分配的平滑程度 (越小越接近硬聚类，越大越均匀)
    :param iterations: Sinkhorn 迭代次数
    :return: 最优传输分配矩阵 (软伪标签), shape: [batch_size, class_num]
    """
    # 转置，shape变为 [class_num, batch_size]
    Q = torch.exp(scores / epsilon).t()

    # 记录批次大小 (B) 和类别数量 (K)
    K, B = Q.shape

    # 避免数值溢出
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for _ in range(iterations):
        # 归一化行: 强制每个簇 (prototype) 能够被均匀分配到样本，避免空簇发生
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # 归一化列: 强制每个样本属于各个簇的概率和为 1
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    # 放大回样本维度的正常概率和，并转置回 [batch_size, class_num]
    Q *= B
    return Q.t()

def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        # 修改后 (加上第五个接收位):
        _, _, xrs, _, _ = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


# def contrastive_train(epoch):
#     tot_loss = 0.
#     mes = torch.nn.MSELoss()
#     for batch_idx, (xs, _, _) in enumerate(data_loader):
#         for v in range(view):
#             xs[v] = xs[v].to(device)
#         optimizer.zero_grad()
#         # 修改后:
#         hs, qs, xrs, zs, prototypes = model(xs)
#         loss_list = []
#         for v in range(view):
#             for w in range(v+1, view):
#                 loss_list.append(criterion.forward_feature(hs[v], hs[w]))
#                 loss_list.append(criterion.forward_label(qs[v], qs[w]))
#             loss_list.append(mes(xs[v], xrs[v]))
#         loss = sum(loss_list)
#         loss.backward()
#         optimizer.step()
#         tot_loss += loss.item()
#     print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))


def contrastive_train(epoch):
    tot_loss = 0.
    mes = torch.nn.MSELoss()

    # 新增超参数：OT 损失的权重，你可以根据数据集在 0.1 到 1.0 之间调优
    lambda_ot = 0.5

    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)

        optimizer.zero_grad()

        # 获取所有输出，包括我们新增的原型 (prototypes)
        hs, qs, xrs, zs, prototypes = model(xs)
        loss_list = []

        for v in range(view):
            # 1. 基础的自编码器重构损失 (防止特征坍缩)
            loss_list.append(mes(xs[v], xrs[v]))

            # 2. 原版的多视图特征 & 标签对比损失 (保留原始论文的跨视图一致性)
            for w in range(v + 1, view):
                loss_list.append(criterion.forward_feature(hs[v], hs[w]))
                loss_list.append(criterion.forward_label(qs[v], qs[w]))

            # 3. 【创新点】最优传输 (OT) 聚类对齐损失
            # 计算当前视图高层特征与原型的余弦相似度
            # 因为 hs[v] 和 prototypes 在网络内部都已经做了 normalize
            scores = torch.matmul(hs[v], prototypes.T)  # [batch_size, class_num]

            # 我们用 detach() 截断梯度，让 target 仅作为监督信号指导当前步
            T_target = sinkhorn_knopp(scores.detach(), epsilon=0.05, iterations=3)

            # 计算交叉熵: 让网络预测的语义标签 qs 逼近最优传输计算出的目标分布 T_target
            # qs[v] 是经过 softmax 的概率输出
            ot_loss = - torch.mean(torch.sum(T_target * torch.log(qs[v] + 1e-8), dim=1))

            # 将 OT 损失加入总损失中
            loss_list.append(lambda_ot * ot_loss)

        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


def make_pseudo_label(model, device):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    model.eval()
    scaler = MinMaxScaler()
    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            hs, _, _, _, _ = model.forward(xs)
        for v in range(view):
            hs[v] = hs[v].cpu().detach().numpy()
            hs[v] = scaler.fit_transform(hs[v])

    kmeans = KMeans(n_clusters=class_num, n_init=100)
    new_pseudo_label = []
    for v in range(view):
        Pseudo_label = kmeans.fit_predict(hs[v])
        Pseudo_label = Pseudo_label.reshape(data_size, 1)
        Pseudo_label = torch.from_numpy(Pseudo_label)
        new_pseudo_label.append(Pseudo_label)

    return new_pseudo_label


def match(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    new_y = np.zeros(y_true.shape[0])
    for i in range(y_pred.size):
        for j in row_ind:
            if y_true[i] == col_ind[j]:
                new_y[i] = row_ind[j]
    new_y = torch.from_numpy(new_y).long().to(device)
    new_y = new_y.view(new_y.size()[0])
    return new_y


def fine_tuning(epoch, new_pseudo_label):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    tot_loss = 0.
    cross_entropy = torch.nn.CrossEntropyLoss()
    for batch_idx, (xs, _, idx) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        # 修改后:
        _, qs, _, _, _ = model(xs)
        loss_list = []
        for v in range(view):
            p = new_pseudo_label[v].numpy().T[0]
            with torch.no_grad():
                q = qs[v].detach().cpu()
                q = torch.argmax(q, dim=1).numpy()
                p_hat = match(p, q)
            loss_list.append(cross_entropy(qs[v], p_hat))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


accs = []
nmis = []
purs = []
if not os.path.exists('./models'):
    os.makedirs('./models')
T = 1
for i in range(T):
    print("ROUND:{}".format(i+1))
    setup_seed(seed)
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
    print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)

    epoch = 1
    while epoch <= args.mse_epochs:
        pretrain(epoch)
        epoch += 1
    while epoch <= args.mse_epochs + args.con_epochs:
        contrastive_train(epoch)
        if epoch == args.mse_epochs + args.con_epochs:
            acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False)
        epoch += 1
    new_pseudo_label = make_pseudo_label(model, device)
    while epoch <= args.mse_epochs + args.con_epochs + args.tune_epochs:
        fine_tuning(epoch, new_pseudo_label)
        if epoch == args.mse_epochs + args.con_epochs + args.tune_epochs:
            acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False)
            state = model.state_dict()
            torch.save(state, './models/' + args.dataset + '.pth')
            print('Saving..')
            accs.append(acc)
            nmis.append(nmi)
            purs.append(pur)
        epoch += 1
