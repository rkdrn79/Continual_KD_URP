import torch
from copy import deepcopy
from argparse import ArgumentParser

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset

import wandb

class Appr(Inc_Learning_Appr):
    """Class implementing the Learning Without Forgetting (LwF) approach
    described in https://arxiv.org/abs/1606.09282
    """

    # Weight decay of 0.0005 is used in the original article (page 4).
    # Page 4: "The warm-up step greatly enhances fine-tuning’s old-task performance, but is not so crucial to either our
    #  method or the compared Less Forgetting Learning (see Table 2(b))."
    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, erf_approach = None, rgr_approach = None, erf_m = 1, rgr_m = 1, cycle_approach = 'all', distill_percent = 0.2,  lamb=1, T=2):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset, erf_approach, rgr_approach, erf_m, rgr_m, cycle_approach, distill_percent)
        self.model_old = None
        self.lamb = lamb
        self.T = T

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Page 5: "lambda is a loss balance weight, set to 1 for most our experiments. Making lambda larger will favor
        # the old task performance over the new task’s, so we can obtain a old-task-new-task performance line by
        # changing lambda."
        parser.add_argument('--lamb', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        # Page 5: "We use T=2 according to a grid search on a held out set, which aligns with the authors’
        #  recommendations." -- Using a higher value for T produces a softer probability distribution over classes.
        parser.add_argument('--T', default=2, type=int, required=False,
                            help='Temperature scaling (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train_loop(self, t, trn_loader, val_loader, tst_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loaderdistill_approach
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader, tst_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader, e):
        """Runs a single epoch"""
        erf_kd_use, rgr_kd_use = self.cycle._get_distill_use(e)

        total_num = 0

        total_total_loss = 0
        total_train_loss = 0
        total_kd_loss = 0
        total_weight = 0

        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            
            # Forward old model
            targets_old = None
            if t > 0 and erf_kd_use: 
                targets_old = self.model_old(images.to(self.device))
            # Forward current model
            outputs = self.model(images.to(self.device))
            total_loss, train_loss, kd_loss, weight = self.criterion(t, outputs, targets.to(self.device), outputs_old = targets_old, epoch = e, erf_kd_use = erf_kd_use, rgr_kd_use = rgr_kd_use)
            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

            #================= Log =================#
            total_num += 1
            total_total_loss += total_loss.item()
            total_train_loss += train_loss.item()
            if t > 0 and erf_kd_use:
                total_kd_loss += kd_loss.item()
                total_weight += weight

        print("| Epoch: ", e + 1, "Loss: ", total_total_loss / total_num, "Train Loss: ", total_train_loss / total_num, "KD Loss: ", total_kd_loss / total_num, "KD Weight: ", total_weight / total_num)
        # Log wandb task t
        wandb.log({"epoch": e,
                "task {} Traing total_loss".format(t): total_total_loss / total_num,
                "task {} Traing train_loss".format(t): total_train_loss / total_num, 
                "task {} Traing kd_loss".format(t): total_kd_loss / total_num, 
                "task {} Traing kd_weight".format(t): total_weight / total_num})

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward old model
                targets_old = None
                if t > 0:
                    targets_old = self.model_old(images.to(self.device))
                # Forward current model
                outputs = self.model(images.to(self.device))
                loss,_,_,_ = self.criterion(t, outputs, targets.to(self.device), targets_old)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                total_num += len(targets)

        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num
    
    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def criterion(self, t, outputs, targets, outputs_old=None, epoch = None, erf_kd_use=False, rgr_kd_use=False):
        """Returns the loss value"""

        total_loss = 0
        train_loss = 0
        kd_loss = 0
        weight = 0

        # loss for current task
        if len(self.exemplars_dataset) > 0:
            train_loss += torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        else:
            train_loss += torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])

        # Knowledge distillation loss for all previous tasks
        if t > 0 and erf_kd_use:
            kd_loss = self.cross_entropy(torch.cat(outputs[:t], dim=1),
                                                torch.cat(outputs_old[:t], dim=1), exp=1.0 / self.T)
            weight = self.erf._get_distill_weight(epoch, train_loss.item(), kd_loss.item())
            kd_loss *= weight * self.lamb

        if t > 0 and rgr_kd_use:
            self.model = self.rgr._get_ema(epoch, self.model)
        total_loss = train_loss + kd_loss

        return total_loss, train_loss, kd_loss, weight

            