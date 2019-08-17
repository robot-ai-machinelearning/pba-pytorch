
import torch
import torch.optim as optim
import torch.nn.functional as F
import os.path as osp
import os
from torchvision.transforms import transforms
from filelock import FileLock
from torchvision import datasets
from torchvision.models import resnet18

try:
    from pba.cifar import CIFAR10
except:
    from cifar import CIFAR10

EPOCH_SIZE = 300
TEST_SIZE = 100


class ModelTrainer(object):

    def __init__(self, config):
        use_cuda = config.get("use_gpu") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.hp_policy = config.get("hp_policy")
            
        self.train_loader, self.test_loader = self.get_data_loaders()
        self.model = resnet18().to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.0005, momentum=0.9)

    def train(self, data_loader):
        self.model.train()
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx * len(data) > EPOCH_SIZE:
                return
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

    def test(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                if batch_idx * len(data) > TEST_SIZE:
                    break
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return correct / total

    def run_train_and_test(self):
        self.train(self.train_loader)
        acc = self.test(self.test_loader)
        return acc

    def save_model(self, ckpt_dir, epoch):
        state = {"model": self.model.state_dict()}
        model_save_name = osp.join(ckpt_dir, f"{epoch}.pth")
        torch.save(state, model_save_name)
        return model_save_name

    def load_model(self, ckpt):
        state = torch.load(ckpt)
        self.model.load_state_dict(state["model"])

    def get_data_loaders(self):
        common_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_loader = torch.utils.data.DataLoader(
            CIFAR10("/home/zhoufan/Code/pba-pytorch/data", self.hp_policy, transform=common_transforms),
            batch_size=32,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            CIFAR10("/home/zhoufan/Code/pba-pytorch/data", self.hp_policy, False, transform=common_transforms),
            batch_size=32, 
            shuffle=True
        )
        return train_loader, test_loader

    def reset_config(self, config):
        self.train_loader.dataset.reset_policy(config.get("hp_policy"))

if __name__ == "__main__":
    config = {
            "hp_policy": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "use_gpu": 1,
        }
    model = ModelTrainer(config)
    model.train(model.train_loader)