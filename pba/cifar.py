try:
    import pba.augmentation_transforms_hp as augmentation_transforms_pba
except:
    import augmentation_transforms_hp as augmentation_transforms_pba
import numpy as np 
import copy
from torch.utils.data import Dataset
import os
import sys
import pickle
from PIL import Image


class CIFAR10(Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, hp_policy, train=True, transform=None):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.transform = transform
        self.policy1 = self.parse_policy(hp_policy[:30], augmentation_transforms_pba)
        self.policy2 = self.parse_policy(hp_policy[30:], augmentation_transforms_pba)

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.data = self.data[:300] if self.train else self.data[:100]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)
        if self.train:
            img = self.apply_policy(img, 32)
            img = img.astype(np.uint8)
        
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def parse_policy(self, policy_emb, augmentation_transforms):
        policy = []
        num_xform = augmentation_transforms.NUM_HP_TRANSFORM
        xform_names = augmentation_transforms.HP_TRANSFORM_NAMES
        assert len(policy_emb
                ) == 2 * num_xform, 'policy was: {}, supposed to be: {}'.format(
                    len(policy_emb), 2 * num_xform)
        for i, xform in enumerate(xform_names):
            policy.append((xform, policy_emb[2 * i] / 10., policy_emb[2 * i + 1]))
        return policy
    
    def apply_policy(self, img, img_size):
        img = augmentation_transforms_pba.apply_policy(self.policy1, img, "cifar10", img_size)
        img = augmentation_transforms_pba.apply_policy(self.policy2, img, "cifar10", img_size)
        return img
    
    def reset_policy(self, hp_policy):
        self.policy1 = self.parse_policy(hp_policy[:30], augmentation_transforms_pba)
        self.policy2 = self.parse_policy(hp_policy[30:], augmentation_transforms_pba)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2
    config = {"hp_policy": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    dataset = CIFAR10("../data", config)
    print(dataset.policy1)
    for i in range(100):
        plt.imshow(dataset[i][0])
        plt.show()