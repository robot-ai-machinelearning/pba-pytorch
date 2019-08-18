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
import tensorflow as tf
import torchvision


def parse_policy(policy_emb, augmentation_transforms):
    policy = []
    num_xform = augmentation_transforms.NUM_HP_TRANSFORM
    xform_names = augmentation_transforms.HP_TRANSFORM_NAMES
    assert len(policy_emb
               ) == 2 * num_xform, 'policy was: {}, supposed to be: {}'.format(
                   len(policy_emb), 2 * num_xform)
    for i, xform in enumerate(xform_names):
        policy.append((xform, policy_emb[2 * i] / 10., policy_emb[2 * i + 1]))
    return policy

def shuffle_data(data, labels):
    """Shuffle data using numpy."""
    np.random.seed(0)
    perm = np.arange(len(data))
    np.random.shuffle(perm)
    data = data[perm]
    labels = labels[perm]
    return data, labels


class SVHN(object):

    def __init__(self, hparams):
        self.hparams = hparams
        self.epochs = 0
        self.curr_train_index = 0

        self.parse_policy(hparams)
        self.load_data(hparams)

        # Apply normalization
        self.train_images = self.train_images.transpose(0, 2, 3, 1) / 255.0
        self.val_images = self.val_images.transpose(0, 2, 3, 1) / 255.0
        self.test_images = self.test_images.transpose(0, 2, 3, 1) / 255.0
        mean = self.train_images.mean(axis=(0, 1, 2))
        std = self.train_images.std(axis=(0, 1, 2))
        self.augmentation_transforms.MEANS["svhn" + '_' + str(hparams.get("train_size"))] = mean
        self.augmentation_transforms.STDS["svhn" + '_' + str(hparams.get("train_size"))] = std
        tf.logging.info('mean:{}    std: {}'.format(mean, std))

        self.train_images = (self.train_images - mean) / std
        self.val_images = (self.val_images - mean) / std
        self.test_images = (self.test_images - mean) / std

        assert len(self.test_images) == len(self.test_labels)
        assert len(self.train_images) == len(self.train_labels)
        assert len(self.val_images) == len(self.val_labels)
        tf.logging.info('train dataset size: {}, test: {}, val: {}'.format(
            len(self.train_images), len(self.test_images), len(self.val_images)))

    def parse_policy(self, hparams):
        self.augmentation_transforms = augmentation_transforms_pba

        raw_policy = hparams.get("hp_policy")
        split = len(raw_policy) // 2
        self.policy = parse_policy(raw_policy[:split], self.augmentation_transforms)
        self.policy.extend(parse_policy(raw_policy[split:], self.augmentation_transforms))
        tf.logging.info("using HP Policy, policy: {}".format(self.policy))

    def load_data(self, hparams):
        # assert hparams.train_size == 1000
        # assert hparams.train_size + hparams.validation_size <= 73257
        train_loader = torchvision.datasets.SVHN(
            root=hparams.get("data_path"), split='train', download=True)
        test_loader = torchvision.datasets.SVHN(
            root=hparams.get("data_path"), split='test', download=True)
        num_classes = 10
        train_data = train_loader.data
        test_data = test_loader.data
        train_labels = train_loader.labels
        test_labels = test_loader.labels

        self.test_images, self.test_labels = test_data, test_labels
        train_data, train_labels = shuffle_data(train_data, train_labels)
        train_size, val_size = hparams.get("train_size"), hparams.get("val_size")
        assert train_size + val_size < 73257
        self.train_images = train_data[:train_size]
        self.train_labels = train_labels[:train_size]
        self.val_images = train_data[-val_size:]
        self.val_labels = train_labels[-val_size:]
        self.num_classes = num_classes
        self.num_train = train_size
        self.image_size = self.train_images.shape[2]
    
    def next_batch(self, iteration=None):
        """Return the next minibatch of augmented data."""
        next_train_index = self.curr_train_index + self.hparams.get("batch_size")
        if next_train_index > self.num_train:
            # Increase epoch number
            epoch = self.epochs + 1
            self.reset()
            self.epochs = epoch
        batched_data = (
            self.train_images[self.curr_train_index:self.curr_train_index +
                              self.hparams.get("batch_size")],
            self.train_labels[self.curr_train_index:self.curr_train_index +
                              self.hparams.get("batch_size")])
        final_imgs = []

        dset = "svhn" + '_' + str(self.hparams.get("train_size"))
        images, labels = batched_data
        for data in images:
            # policy schedule
            final_img = self.augmentation_transforms.apply_policy(
                self.policy,
                data,
                "cifar10",
                dset,
                image_size=self.image_size)
            final_imgs.append(final_img)
        batched_data = (np.array(final_imgs, np.float32).transpose(0, 3, 1, 2), labels)
        self.curr_train_index += self.hparams.get("batch_size")
        return batched_data
    
    def reset(self):
        """Reset training data and index into the training data."""
        self.epochs = 0
        # Shuffle the training data
        perm = np.arange(self.num_train)
        np.random.shuffle(perm)
        assert self.num_train == self.train_images.shape[0], 'Error incorrect shuffling mask'
        self.train_images = self.train_images[perm]
        self.train_labels = self.train_labels[perm]
        self.curr_train_index = 0
    
    def reset_policy(self, new_hparams):
        self.hparams = new_hparams
        self.parse_policy(new_hparams)
        return


if __name__ == "__main__":
    hparams = {
        "data_path": "/home/zhoufan/Code/pba-pytorch/data",
        "train_size": 1000,
        "val_size": 7325,
        "hp_policy": [0] * 4 * augmentation_transforms_pba.NUM_HP_TRANSFORM,
        "batch_size": 128,
        }
    data = SVHN(hparams)
    data.next_batch()