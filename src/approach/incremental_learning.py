import time
import torch
import numpy as np
from argparse import ArgumentParser

from loggers.exp_logger import ExperimentLogger
from datasets.exemplars_dataset import ExemplarsDataset
from distill_approach.ERF import ERF
from distill_approach.RGR import RGR
from distill_approach.CYCLE import CYCLE
from distill_approach.AKD import AKD

import wandb
import os

class Inc_Learning_Appr:
    """Basic class for implementing incremental learning approaches"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger: ExperimentLogger = None, exemplars_dataset: ExemplarsDataset = None,
                 erf_approach: ERF = None, rgr_approach: RGR = None, erf_m = 1, rgr_m = 1, cycle_approach = 'all', distill_percent = 0.2):
        self.model = model
        self.device = device
        self.nepochs = nepochs
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.wd = wd
        self.multi_softmax = multi_softmax
        self.logger = logger
        self.exemplars_dataset = exemplars_dataset
        self.warmup_epochs = wu_nepochs
        self.warmup_lr = lr * wu_lr_factor
        self.warmup_loss = torch.nn.CrossEntropyLoss()
        self.fix_bn = fix_bn
        self.eval_on_train = eval_on_train
        self.optimizer = None
        self.erf_approach = erf_approach
        self.rgr_approach = rgr_approach
        self.erf_m = erf_m
        self.rgr_m = rgr_m
        self.cycle_approach = cycle_approach
        self.distill_percent = distill_percent

        if 'adaptive' in self.erf_approach:
            self.erf = AKD(akd_approach = self.erf_approach, total_epochs = self.nepochs, m = self.erf_m)
        else:
            self.erf = ERF(erf_approach = self.erf_approach, m = self.erf_m)
        self.rgr = RGR(rgr_approach = self.rgr_approach, m = self.rgr_m)
        self.cycle = CYCLE(cycle_approach = self.cycle_approach, train_epochs = self.nepochs, distill_percent = self.distill_percent)

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        return parser.parse_known_args(args)

    @staticmethod
    def exemplars_dataset_class():
        """Returns a exemplar dataset to use during the training if the approach needs it
        :return: ExemplarDataset class or None
        """
        return None

    def _get_optimizer(self):
        """Returns the optimizer"""
        return torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train(self, t, trn_loader, val_loader, tst_loader):
        """Main train structure"""
        self.pre_train_process(t, trn_loader)

        """
        network = "resnet32" # < -- Change this to the network name
        if os.path.exists("./result/task0_model/{}.pth".format(network)) and t == 0:
            self.model.load_state_dict(torch.load("./result/task0_model/{}.pth".format(network)),strict = False)
            print("Load model from ./result/task0_model/{}.pth".format(network))
            print("Skip training task 0")
        else:
            self.train_loop(t, trn_loader, val_loader, tst_loader)
            torch.save(self.model.state_dict(), "./result/task0_model/{}.pth".format(network))
            
        """
        self.train_loop(t, trn_loader, val_loader, tst_loader)
        self.post_train_process(t, trn_loader)

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""

        # Warm-up phase
        if self.warmup_epochs and t > 0:
            self.optimizer = torch.optim.SGD(self.model.heads[-1].parameters(), lr=self.warmup_lr)
            # Loop epochs -- train warm-up head
            for e in range(self.warmup_epochs):
                warmupclock0 = time.time()
                self.model.heads[-1].train()
                for images, targets in trn_loader:
                    outputs = self.model(images.to(self.device))
                    loss = self.warmup_loss(outputs[t], targets.to(self.device) - self.model.task_offset[t])
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.heads[-1].parameters(), self.clipgrad)
                    self.optimizer.step()
                warmupclock1 = time.time()
                with torch.no_grad():
                    total_loss, total_acc_taw = 0, 0
                    self.model.eval()
                    for images, targets in trn_loader:
                        outputs = self.model(images.to(self.device))
                        loss = self.warmup_loss(outputs[t], targets.to(self.device) - self.model.task_offset[t])
                        pred = torch.zeros_like(targets.to(self.device))
                        for m in range(len(pred)):
                            this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
                            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
                        hits_taw = (pred == targets.to(self.device)).float()
                        total_loss += loss.item() * len(targets)
                        total_acc_taw += hits_taw.sum().item()
                total_num = len(trn_loader.dataset.labels)
                trn_loss, trn_acc = total_loss / total_num, total_acc_taw / total_num
                warmupclock2 = time.time()
                print('| Warm-up Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, warmupclock1 - warmupclock0, warmupclock2 - warmupclock1, trn_loss, 100 * trn_acc))
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=trn_loss, group="warmup")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * trn_acc, group="warmup")

    def train_loop(self, t, trn_loader, val_loader, tst_loader):
        """Contains the epochs loop"""
        lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
        best_model = self.model.get_copy()
        self.optimizer = self._get_optimizer()
        self.rgr._reset_eam(t)
        
        if 'adaptive' in self.erf_approach:
            self.erf._reset_save(t)

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(t, trn_loader, e)
            clock1 = time.time()
            
            # Adaptive method plus save the distill weight
            if 'adaptive' in self.erf_approach:
                valid_total_loss, valid_train_loss, valid_kd_loss, valid_tag_acc, valid_taw_acc = self.eval(t, val_loader)
                self.erf._save_distill_weight(valid_total_loss = valid_total_loss, valid_train_loss = valid_train_loss, valid_kd_loss = valid_kd_loss, valid_taw_accuracie = valid_taw_acc, valid_tag_accuracies = valid_tag_acc, epoch = e)

            if self.eval_on_train:
                total_loss, _, _, train_acc, _ = self.eval(t, trn_loader)
                clock2 = time.time()

                erf_kd_use, rgr_kd_use = self.cycle._get_distill_use(e)
                if t == 0:
                    erf_kd_use = False
                    rgr_kd_use = False
                    
                print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: total loss={:.3f}, TAw acc={:5.1f}%  | ERF KD : {} , RGR KD : {}|'.format(
                    e + 1, clock1 - clock0, clock2 - clock1, total_loss, 100 * train_acc, erf_kd_use, rgr_kd_use), end='')
                
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=total_loss, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")

                wandb.log({"epoch" : e,
                    "task "+ str(t) +" total_loss": total_loss,
                    "task "+ str(t) +" train_acc": 100 * train_acc})

            else:
                print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0))

            # Valid
            clock3 = time.time()
            valid_loss,_,_,valid_acc, _ = self.eval(t, val_loader)
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc))
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")
            wandb.log({"epoch" : e ,
                        "task "+ str(t) +" valid_loss": valid_loss, 
                       "task "+ str(t) +" valid_acc": 100 * valid_acc})

            if self.eval_on_train and t > 0:
                # Test
                for u in range(t):
                    test_loss,_,_, test_acc_taw, test_acc_tag = self.eval(u, tst_loader[u])
                    print('| Epoch : {}, train task {}, test_acc_taw {}, test_acc_tag {}'.format(e+1, u, test_acc_taw, test_acc_tag))
                    wandb.log({"epoch" : e ,
                        "train "+str(t)+" task"+ str(u) +" test_acc_taw": test_acc_taw, 
                        "train"+str(t)+" task"+ str(u) +" test_acc_tag": test_acc_tag})

            # Adapt learning rate - patience scheme - early stopping regularization
            if valid_loss < best_loss:
                # if the loss goes down, keep it as the best model and end line with a star ( * )
                best_loss = valid_loss
                best_model = self.model.get_copy()
                patience = self.lr_patience
                print(' *', end='')

            else:
                # if the loss does not go down, decrease patience
                patience -= 1
                if patience <= 0:
                    # if it runs out of patience, reduce the learning rate
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        # if the lr decreases below minimum, stop the training session
                        print()
                        lr = self.lr_min
                    # reset patience and recover best model so far to continue training
                    patience = self.lr_patience
                    self.optimizer.param_groups[0]['lr'] = lr
                    self.model.set_state_dict(best_model)
            self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
            print()
            
        self.model.set_state_dict(best_model)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        pass

    def train_epoch(self, t, trn_loader, e):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            # Forward current model
            outputs = self.model(images.to(self.device))
            total_loss, train_loss, kd_loss, weight = self.criterion(t, outputs, targets.to(self.device), e)
            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward current model
                outputs = self.model(images.to(self.device))
                loss,_,_,_= self.criterion(t, outputs, targets.to(self.device))
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)

        return total_loss / total_num, total_loss / total_num, 0, total_acc_taw / total_num, total_acc_tag / total_num

    def calculate_metrics(self, outputs, targets):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        pred = torch.zeros_like(targets.to(self.device))
        # Task-Aware Multi-Head
        for m in range(len(pred)):
            this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
        hits_taw = (pred == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        if self.multi_softmax:
            outputs = [torch.nn.functional.log_softmax(output, dim=1) for output in outputs]
            pred = torch.cat(outputs, dim=1).argmax(1)
        else:
            pred = torch.cat(outputs, dim=1).argmax(1)
        hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag

    def criterion(self, t, outputs, targets, e):
        """Returns the loss value"""
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
