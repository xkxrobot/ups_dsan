import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from .augmentations import RandAugment, CutoutRandom
import pickle
import os


def get_cifar10(root='data/datasets', n_lbl=4000, ssl_idx=None, pseudo_lbl=None, itr=0, split_txt=''):
    """将划分为有标签和无标签数据的数据集，以及伪标签，负标签，作为获取数据的入口"""
    os.makedirs(root, exist_ok=True) #create the root directory for saving data
    # augmentations
    transform_train = transforms.Compose([
        RandAugment(3,4),  #from https://arxiv.org/pdf/1909.13719.pdf. For CIFAR-10 M=3, N=4
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
        CutoutRandom(n_holes=1, length=16, random=True)
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
    ])
    # 半监督训练的数据，包括有标签数据和无标签数据，base_dataset表示所有图片对应的类别序列，是一个一维向量 1x50000
    if ssl_idx is None:
        base_dataset = datasets.CIFAR10(root, train=True, download=True)
        train_lbl_idx, train_unlbl_idx = lbl_unlbl_split(base_dataset.targets, n_lbl, 10)
        os.makedirs('data/splits', exist_ok=True)
        f = open(os.path.join('data/splits', f'cifar10_basesplit_{n_lbl}_{split_txt}.pkl'),"wb")
        lbl_unlbl_dict = {'lbl_idx': train_lbl_idx, 'unlbl_idx': train_unlbl_idx}
        pickle.dump(lbl_unlbl_dict,f)
    # 已经存在半监督数据
    else:
        lbl_unlbl_dict = pickle.load(open(ssl_idx, 'rb'))
        train_lbl_idx = lbl_unlbl_dict['lbl_idx']
        train_unlbl_idx = lbl_unlbl_dict['unlbl_idx']

    lbl_idx = train_lbl_idx
    # 挑选伪标签中哪些作为负标签，哪些作为负标签，哪些不选择作为下一次训练的无标签数据
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

        #balance the labeled and unlabeled data 
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

    train_lbl_dataset = CIFAR10SSL(
        root, lbl_idx, train=True, transform=transform_train,
        pseudo_idx=pseudo_idx, pseudo_target=pseudo_target,
        nl_idx=nl_idx, nl_mask=nl_mask)
    
    if nl_idx is not None:
        train_nl_dataset = CIFAR10SSL(
            root, np.array(nl_idx), train=True, transform=transform_train,
            pseudo_idx=pseudo_idx, pseudo_target=pseudo_target,
            nl_idx=nl_idx, nl_mask=nl_mask)

    train_unlbl_dataset = CIFAR10SSL(
    root, train_unlbl_idx, train=True, transform=transform_val)

    test_dataset = datasets.CIFAR10(root, train=False, transform=transform_val, download=True)

    if nl_idx is not None:
        return train_lbl_dataset, train_nl_dataset, train_unlbl_dataset, test_dataset
    else:
        return train_lbl_dataset, train_unlbl_dataset, train_unlbl_dataset, test_dataset


def get_cifar100(root='data/datasets', n_lbl=10000, ssl_idx=None, pseudo_lbl=None, itr=0, split_txt=''):
    ## augmentations
    transform_train = transforms.Compose([
        RandAugment(3,4),  #from https://arxiv.org/pdf/1909.13719.pdf. For CIFAR-10 M=3, N=4
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
        CutoutRandom(n_holes=1, length=16, random=True)
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ])

    if ssl_idx is None:
        base_dataset = datasets.CIFAR100(root, train=True, download=True)
        train_lbl_idx, train_unlbl_idx = lbl_unlbl_split(base_dataset.targets, n_lbl, 100)
        
        f = open(os.path.join('data/splits', f'cifar100_basesplit_{n_lbl}_{split_txt}.pkl'),"wb")
        lbl_unlbl_dict = {'lbl_idx': train_lbl_idx, 'unlbl_idx': train_unlbl_idx}
        pickle.dump(lbl_unlbl_dict,f)
    
    else:
        lbl_unlbl_dict = pickle.load(open(ssl_idx, 'rb'))
        train_lbl_idx = lbl_unlbl_dict['lbl_idx']
        train_unlbl_idx = lbl_unlbl_dict['unlbl_idx']

    lbl_idx = train_lbl_idx
    if pseudo_lbl is not None:
        pseudo_lbl_dict = pickle.load(open(pseudo_lbl, 'rb'))
        pseudo_idx = pseudo_lbl_dict['pseudo_idx']
        pseudo_target = pseudo_lbl_dict['pseudo_target']
        nl_idx = pseudo_lbl_dict['nl_idx']
        nl_mask = pseudo_lbl_dict['nl_mask']
        lbl_idx = np.array(lbl_idx + pseudo_idx)

        #balance the labeled and unlabeled data 
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

    train_lbl_dataset = CIFAR100SSL(
        root, lbl_idx, train=True, transform=transform_train,
        pseudo_idx=pseudo_idx, pseudo_target=pseudo_target,
        nl_idx=nl_idx, nl_mask=nl_mask)
    
    if nl_idx is not None:
        train_nl_dataset = CIFAR100SSL(
            root, np.array(nl_idx), train=True, transform=transform_train,
            pseudo_idx=pseudo_idx, pseudo_target=pseudo_target,
            nl_idx=nl_idx, nl_mask=nl_mask)

    train_unlbl_dataset = CIFAR100SSL(
    root, train_unlbl_idx, train=True, transform=transform_val)

    test_dataset = datasets.CIFAR100(root, train=False, transform=transform_val, download=True)

    if nl_idx is not None:
        return train_lbl_dataset, train_nl_dataset, train_unlbl_dataset, test_dataset
    else:
        return train_lbl_dataset, train_unlbl_dataset, train_unlbl_dataset, test_dataset

# 划分有标签和无标签的数据
def lbl_unlbl_split(lbls, n_lbl, n_class):
    """划分有标签和无标签数据，索引上的划分"""
    # 每个类选择多少个有标签数据
    lbl_per_class = n_lbl // n_class
    lbls = np.array(lbls)
    lbl_idx = []
    unlbl_idx = []
    for i in range(n_class):
        # np.where(lbls == i)[0]表示二维列表中的行索引,属于某一个类别的所有下标
        idx = np.where(lbls == i)[0]
        # 打乱每一个类别中的序号
        np.random.shuffle(idx)
        # 扩展有标签数据
        lbl_idx.extend(idx[:lbl_per_class])
        # 扩展无标签数据
        unlbl_idx.extend(idx[lbl_per_class:])
    return lbl_idx, unlbl_idx


# 使得索引数据对应于图片数据，对选择为有标签的数据进行数据增强，返回数据加载器loader
class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True, pseudo_idx=None, pseudo_target=None,
                 nl_idx=None, nl_mask=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        
        self.targets = np.array(self.targets)
        # nl_mask的shape => (50000, 10)
        self.nl_mask = np.ones((len(self.targets), len(np.unique(self.targets))))
        
        if nl_mask is not None:
            self.nl_mask[nl_idx] = nl_mask

        if pseudo_target is not None:
            self.targets[pseudo_idx] = pseudo_target

        if indexs is not None:
            # 表示对应有标签数据，无标签数据，伪标签数据的索引
            indexs = np.array(indexs)
            # data表示图片数据，此处表示50000张图片数据中提取出符合indexs序号的图片数据
            self.data = self.data[indexs]
            # targets表示图片数据对应的类别编号,此处表示50000张图片中符合下标indexs的target序号提取出来
            self.targets = np.array(self.targets)[indexs]
            # 从全部数据中屏蔽一些标签，将有标签数据变为无标签数据,每一个数据的mask向量表示为列表[1,] * 10
            self.nl_mask = np.array(self.nl_mask)[indexs]
            # indexs表示图片所在列表的下标序号
            self.indexs = indexs
        else:
            # 如果没有给出图片所在的下标数组，则通过所给的标签序号列表获取图片所在的下标数组
            self.indexs = np.arange(len(self.targets))
        

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # 将数组数据导出为图像数据
        img = Image.fromarray(img)

        if self.transform is not None:
            # 对图像数据进行数据增强
            img = self.transform(img)

        if self.target_transform is not None:
            # 对标签数据进行数据增强
            target = self.target_transform(target)
        # 返回数据增强后的图片数据和相对应的标签序号数据
        return img, target, self.indexs[index], self.nl_mask[index]


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True, pseudo_idx=None, pseudo_target=None,
                 nl_idx=None, nl_mask=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        
        self.targets = np.array(self.targets)
        self.nl_mask = np.ones((len(self.targets), len(np.unique(self.targets))))
        
        if nl_mask is not None:
            self.nl_mask[nl_idx] = nl_mask

        if pseudo_target is not None:
            self.targets[pseudo_idx] = pseudo_target

        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.nl_mask = np.array(self.nl_mask)[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))
        

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target, self.indexs[index], self.nl_mask[index]