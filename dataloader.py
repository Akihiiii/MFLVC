from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch


class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()


class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'label.npy')

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(
           x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


# class MultiViewDataset(Dataset):
#     """
#     通用多视图数据集加载类
#     适用于 Mfeat, UCI-3view, COIL20, ALOI-100 等数据集
#     数据格式：mat文件包含 'fea' (1, n_views) 和 'gt' (n_samples, 1)
#     """
#     def __init__(self, path, dataset_name, n_views):
#         data = scipy.io.loadmat(path + dataset_name)
#
#         # 加载多视图特征
#         fea = data['fea']  # shape: (1, n_views)
#         self.views = []
#         for i in range(n_views):
#             view_data = fea[0, i].astype(np.float32)
#             self.views.append(view_data)
#
#         # 加载标签 (MATLAB标签从1开始，转换为从0开始)
#         self.labels = data['gt'].astype(np.int32).reshape(-1) - 1
#         self.n_views = n_views
#
#     def __len__(self):
#         return self.views[0].shape[0]
#
#     def __getitem__(self, idx):
#         views = [torch.from_numpy(view[idx]) for view in self.views]
#         return *views, torch.from_numpy(np.array(self.labels[idx])), torch.from_numpy(np.array(idx)).long()


class MultiViewDataset3(Dataset):
    """
    三视图数据集
    适用于 fea: (1, 3)
    """

    def __init__(self, path, dataset_name):
        data = scipy.io.loadmat(path + dataset_name)

        fea = data['fea']  # shape: (1, 3)

        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(fea[0, 0].astype(np.float32))
        self.view2 = scaler.fit_transform(fea[0, 1].astype(np.float32))
        self.view3 = scaler.fit_transform(fea[0, 2].astype(np.float32))

        # 标签从1开始 → 改成0开始
        labels = data['gt'].astype(np.int32).reshape(-1) - 1
        self.labels = labels

    def __len__(self):
        return self.view1.shape[0]

    def __getitem__(self, idx):
        return [
            torch.from_numpy(self.view1[idx]),
            torch.from_numpy(self.view2[idx]),
            torch.from_numpy(self.view3[idx])
        ], torch.from_numpy(np.array(self.labels[idx])), torch.from_numpy(np.array(idx)).long()

class MultiViewDataset4(Dataset):
    """
    四视图数据集
    适用于 fea: (1, 4)
    """

    def __init__(self, path, dataset_name):
        data = scipy.io.loadmat(path + dataset_name)

        fea = data['fea']  # shape: (1, 4)

        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(fea[0, 0].astype(np.float32))
        self.view2 = scaler.fit_transform(fea[0, 1].astype(np.float32))
        self.view3 = scaler.fit_transform(fea[0, 2].astype(np.float32))
        self.view4 = scaler.fit_transform(fea[0, 3].astype(np.float32))

        labels = data['gt'].astype(np.int32).reshape(-1) - 1
        self.labels = labels

    def __len__(self):
        return self.view1.shape[0]

    def __getitem__(self, idx):
        return [
            torch.from_numpy(self.view1[idx]),
            torch.from_numpy(self.view2[idx]),
            torch.from_numpy(self.view3[idx]),
            torch.from_numpy(self.view4[idx])
        ], torch.from_numpy(np.array(self.labels[idx])), torch.from_numpy(np.array(idx)).long()

def load_data(dataset):
    if dataset == "BDGP":
        dataset = BDGP('./data/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "CCV":
        dataset = CCV('./data/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "Fashion":
        dataset = Fashion('./data/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "Caltech-2V":
        dataset = Caltech('data/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech('data/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech('data/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech('data/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    elif dataset == "Mfeat":
        dataset = MultiViewDataset3('./data/', 'Mfeat.mat')
        dims = [216, 76, 47]
        view = 3
        data_size = 2000
        class_num = 10
    elif dataset == "UCI":
        dataset = MultiViewDataset3('./data/', 'UCI-3view.mat')
        dims = [240, 76, 6]
        view = 3
        data_size = 2000
        class_num = 10
    elif dataset == "COIL20":
        dataset = MultiViewDataset3('./data/', 'COIL20.mat')
        dims = [1024, 3304, 6750]
        view = 3
        data_size = 1440
        class_num = 20
    elif dataset == "ALOI-100":
        dataset = MultiViewDataset4('./data/', 'ALOI-100.mat')
        dims = [77, 13, 64, 125]
        view = 4
        data_size = 10800
        class_num = 100
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num


