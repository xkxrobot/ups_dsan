import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
import pickle
import os

# 3250 = 65(类别) * 50(同类别图片数量)
def get_office_home(root_path='data/datasets', n_lbl=930, ssl_idx=None, pseudo_lbl=None, itr=0, split_txt=''):
    transform_train = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
    ])
    directory = 'Original_images/amazon/images'
    # 先不进行transform数据增强，首先选择出标签数据，然后对标签数据进行数据增加训练
    base_dataset = datasets.ImageFolder(root=os.path.join(root_path, directory))

    # 半监督训练的数据，包括有标签数据和无标签数据
    if ssl_idx is None:

        train_lbl_idx, train_unlbl_idx = lbl_unlbl_split(base_dataset.targets, n_lbl, 31)

        os.makedirs('data/splits', exist_ok=True)
        f = open(os.path.join('data/splits', f'office_home_basesplit_{n_lbl}_{split_txt}.pkl'), "wb")
        # 将划分好的索引数据保存到文件中，方便下次导入
        lbl_unlbl_dict = {'lbl_idx': train_lbl_idx, 'unlbl_idx': train_unlbl_idx}
        pickle.dump(lbl_unlbl_dict, f)
    # 已经存在半监督数据
    else:
        lbl_unlbl_dict = pickle.load(open(ssl_idx, 'rb'))
        train_lbl_idx = lbl_unlbl_dict['lbl_idx']
        train_unlbl_idx = lbl_unlbl_dict['unlbl_idx']
    # 获取训练集中的有标签数据的索引数据列表
    lbl_idx = train_lbl_idx

    # 挑选伪标签中哪些作为正标签，哪些作为负标签，哪些不选择作为下一次训练的无标签数据
    if pseudo_lbl is not None:
        # 从文件中加载保存的伪标签数据
        pseudo_lbl_dict = pickle.load(open(pseudo_lbl, 'rb'))
        # 提取伪标签的索引
        pseudo_idx = pseudo_lbl_dict['pseudo_idx']
        # 伪标签的目标
        pseudo_target = pseudo_lbl_dict['pseudo_target']
        # 伪标签中的负标签
        nl_idx = pseudo_lbl_dict['nl_idx']
        nl_mask = pseudo_lbl_dict['nl_mask']
        # 伪标签中的可以作为正标签的数据，与源标签数据合并
        lbl_idx = np.array(lbl_idx + pseudo_idx)

        # balance the labeled and unlabeled data
        if len(nl_idx) > len(lbl_idx):
            exapand_labeled = len(nl_idx) // len(lbl_idx)
            lbl_idx = np.hstack([lbl_idx for _ in range(exapand_labeled)])

            if len(lbl_idx) < len(nl_idx):
                diff = len(nl_idx) - len(lbl_idx)
                lbl_idx = np.hstack((lbl_idx, np.random.choice(lbl_idx, diff)))
            else:
                assert len(lbl_idx) == len(nl_idx)
    else:
        pseudo_idx = None
        pseudo_target = None
        nl_idx = None
        nl_mask = None

    train_lbl_dataset = OfficeHomeSsl(
        root_path, lbl_idx, base_dataset, train=True, transform=transform_train,
        pseudo_idx=pseudo_idx, pseudo_target=pseudo_target,
        nl_idx=nl_idx, nl_mask=nl_mask)

    if nl_idx is not None:
        train_nl_dataset = OfficeHomeSsl(
            root_path, np.array(nl_idx), base_dataset, train=True, transform=transform_train,
            pseudo_idx=pseudo_idx, pseudo_target=pseudo_target,
            nl_idx=nl_idx, nl_mask=nl_mask)

    train_unlbl_dataset = OfficeHomeSsl(
        root_path, train_unlbl_idx, base_dataset, train=True, transform=transform_val)

    # 获取测试集
    test_dataset = datasets.CIFAR10(root_path, train=False, transform=transform_val, download=True)

    if nl_idx is not None:
        return train_lbl_dataset, train_nl_dataset, train_unlbl_dataset, test_dataset
    else:
        return train_lbl_dataset, train_unlbl_dataset, train_unlbl_dataset, test_dataset

def lbl_unlbl_split(lbls, n_lbl, n_class):
    """
    对标签数据源lbls进行数据划分，划分为有标签数据和无标签数据
    return -> 划分后的数据序号列表
    """
    # 每个类选择多少个有标签数据
    lbl_per_class = n_lbl // n_class
    lbls = np.array(lbls)
    lbl_idx = []
    unlbl_idx = []
    for i in range(n_class):
        # idx表示标签为i的横坐标，进一步表示标签为i的所有图片的序号
        idx = np.where(lbls == i)[0]
        # 打乱每一个类别中的序号
        np.random.shuffle(idx)
        # 扩展有标签数据，选择乱序后的前lel_per_class个图片作为第i类的有标签数据
        lbl_idx.extend(idx[:lbl_per_class])
        # 扩展无标签数据，选择乱序后的剩余图片作为第i类的无标签数据
        unlbl_idx.extend(idx[lbl_per_class:])
    # 返回有标签数据的图片索引列表，无标签数据的图片索引列表
    return lbl_idx, unlbl_idx

class OfficeHomeSsl(Dataset):
    def __init__(self, root, indexs,dataset,train=True,
                 transform=None, target_transform=None,
                 pseudo_idx=None, pseudo_target=None,
                 nl_idx=None, nl_mask=None):
        self.target_transform = target_transform
        self.transform = transform
        self.train = train
        self.root = root

        self.targets = np.array(dataset.targets)
        # nl_mask的shape => (50000, 10)
        self.nl_mask = np.ones((len(self.targets), len(np.unique(self.targets))))

        if nl_mask is not None:
            self.nl_mask[nl_idx] = nl_mask

        if pseudo_target is not None:
            self.targets[pseudo_idx] = pseudo_target

        if indexs is not None:
            # 表示对应有标签数据，无标签数据，伪标签数据的索引
            indexs = np.array(indexs)
            # data表示图片数据
            self.data = np.array(dataset.imgs)[indexs]
            # targets表示图片数据对应的类别编号
            self.targets = np.array(self.targets)[indexs]
            # 屏蔽一些标签，将有标签数据变为无标签数据,每一个数据的mask向量表示为列表[1,] * 10
            self.nl_mask = np.array(self.nl_mask)[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        with Image.open(img[0]) as img:
            img = self.transform(img)

        if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target, self.indexs[index], self.nl_mask[index]

    def __len__(self) -> int:
        return len(self.data)