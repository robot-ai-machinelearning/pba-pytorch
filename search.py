
from pba.model import ModelTrainer
from ray.tune import run_experiments, Trainable
import ray
from ray.tune.schedulers import PopulationBasedTraining
import ray.tune as tune
import numpy as np
import random


class RayModel(Trainable):

    def _setup(self, *args):
        self.trainer = ModelTrainer(self.config)

    def _train(self):
        train_acc, val_acc = self.trainer.run_model(self._iteration)
        return {"train_acc": train_acc, "val_acc": val_acc}
    
    def _save(self, ckpt_dir):
        save_name = self.trainer.save_model(ckpt_dir, self._iteration)
        return save_name
    
    def _restore(self, ckpt):
        self.trainer.load_model(ckpt)
    
    def reset_config(self, new_config):
        self.config = new_config
        self.trainer.reset_config(self.config)
        return True

def main():
    train_spec = {
        "run": RayModel,
        "resources_per_trial": {
            "cpu": 2,
            "gpu": 0.15,
        },
        "stop": {
            "training_iteration": 200,
        },
        "config": {
            "data_path": "/home/zhoufan/Code/pba-pytorch/data",
            "train_size": 1000,
            "val_size": 7325,
            "hp_policy": [0] * 60,
            "batch_size": 128,
        },
        "local_dir": "results",
        "checkpoint_freq": 0,
        "num_samples": 16,
    }

    def explore(config):
        new_params = []
        for i, param in enumerate(config["hp_policy"]):
            if random.random() < 0.2:
                if i % 2 == 0:
                    new_params.append(random.randint(0, 10))
                else:
                    new_params.append(random.randint(0, 9))
            else:
                amt = np.random.choice(
                    [0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25])
                # Cast np.int64 to int for py3 json
                amt = int(amt)
                if random.random() < 0.5:
                    new_params.append(max(0, param - amt))
                else:
                    if i % 2 == 0:
                        new_params.append(min(10, param + amt))
                    else:
                        new_params.append(min(9, param + amt))
        config["hp_policy"] = new_params
        return config

    ray.init()

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        reward_attr="val_acc",
        perturbation_interval=5,
        # hyperparam_mutations={
        #     "lr": lambda: random.uniform(0.0001, 0.02),
        #     "momentum": [0.5, 0.6, 0.7, 0.9],
        # }
        custom_explore_fn=explore,
    )

    anas = run_experiments(
        {
            "svhn": train_spec,
        },
        scheduler=pbt,
        reuse_actors=True,
        verbose=False,
    )

main()
    