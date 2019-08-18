
import torch
import torch.optim as optim
import torch.nn.functional as F
import os.path as osp
import os
from torchvision.transforms import transforms
from filelock import FileLock
from torchvision import datasets
from torchvision.models import resnet18
import numpy as np

try:
    from pba.cifar import CIFAR10
    from pba.svhn import SVHN
except:
    from cifar import CIFAR10
    from svhn import SVHN

EPOCH_SIZE = 300
TEST_SIZE = 100


class ModelTrainer(object):

    def __init__(self, hparams):
        self.hparams = hparams

        np.random.seed(0)
        self.data_loader = SVHN(hparams) 
        np.random.seed()
        self.data_loader.reset()   

        self.model = resnet18(num_classes=10).cuda()

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def _run_train_loop(self, curr_epoch):
        self.model.train()
        correct = 0.
        total = 0.
        steps_per_epoch = int(self.hparams.get("train_size") / self.hparams.get("batch_size"))
        for step in range(steps_per_epoch):
            train_images, train_labels = self.data_loader.next_batch()
            train_images, train_labels = torch.from_numpy(train_images), torch.from_numpy(train_labels)
            train_images, train_labels = train_images.cuda(), train_labels.cuda()
            self.optimizer.zero_grad()
            logits = self.model(train_images)
            loss = F.cross_entropy(logits, train_labels)
            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(logits.data, 1)
            total += train_labels.size(0)
            correct += (predicted == train_labels).sum().item()
        return correct / total

    def _run_test_loop(self):
        self.model.eval()
        images, labels = self.data_loader.val_images, self.data_loader.val_labels
        steps_per_epoch = int(self.hparams.get("val_size") / self.hparams.get("batch_size"))
        correct = 0.
        total = 0.
        with torch.no_grad():
            for step in range(steps_per_epoch):
                test_images = images[step * self.hparams.get("batch_size") : (step + 1) * self.hparams.get("batch_size")]
                test_labels = labels[step * self.hparams.get("batch_size") : (step + 1) * self.hparams.get("batch_size")]
                test_images = test_images.transpose(0, 3, 1, 2).astype(np.float32)
                test_images, test_labels = torch.from_numpy(test_images).cuda(), torch.from_numpy(test_labels).cuda()
                logits = self.model(test_images)
                _, predicted = torch.max(logits.data, 1)
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()
        return correct / total
            

    def run_model(self, epoch):
        train_acc = self._run_train_loop(epoch)
        val_acc = self._run_test_loop()
        return train_acc, val_acc

    def save_model(self, ckpt_dir, epoch):
        state = {"model": self.model.state_dict()}
        model_save_name = osp.join(ckpt_dir, f"{epoch}.pth")
        torch.save(state, model_save_name)
        return model_save_name

    def load_model(self, ckpt):
        state = torch.load(ckpt)
        self.model.load_state_dict(state["model"])

    def reset_config(self, new_hparams):
        self.hparams = new_hparams 
        self.data_loader.reset_policy(new_hparams)
        return

if __name__ == "__main__":
    hparams = {
        "data_path": "/home/zhoufan/Code/pba-pytorch/data",
        "train_size": 1000,
        "val_size": 7325,
        "hp_policy": [0] * 4 * 15,
        "batch_size": 32,
        }
    model = ModelTrainer(hparams)
    model._run_test_loop()