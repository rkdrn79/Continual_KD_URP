class AKD():
    def __init__(self, akd_approach, total_epochs, m = 2):
        self.akd_approach = akd_approach
        self.m = m
        self.total_epochs = total_epochs

        self.total_loss = [0]*total_epochs
        self.train_loss = [0]*total_epochs
        self.kd_loss = [0]*total_epochs

        self.valid_total_loss = [0]*total_epochs
        self.valid_train_loss = [0]*total_epochs
        self.valid_kd_loss = [0]*total_epochs
        self.valid_taw_accuracies = [0]*total_epochs
        self.valid_tag_accuracies = [0]*total_epochs

        self.current_task = 0

    def _get_distill_weight(self, epoch):
        return self.m * self._make_adaptive_distll_weight(epoch) 

    def _save_distill_weight(self, total_loss = None, train_loss = None, kd_loss = None, valid_total_loss = None, valid_train_loss = None, valid_kd_loss = None, valid_taw_accuracie = None, valid_tag_accuracies = None,  epoch = 0):
        # save the training loss
        if total_loss is not None:
            self.total_loss[epoch] += total_loss
        if train_loss is not None:
            self.train_loss[epoch] += train_loss
        if kd_loss is not None:
            self.kd_loss[epoch] += kd_loss

        # save the validation loss
        if valid_total_loss is not None:
            self.valid_total_loss[epoch] += valid_total_loss
        if valid_train_loss is not None:
            self.valid_train_loss[epoch] += valid_train_loss
        if valid_kd_loss is not None:
            self.valid_kd_loss[epoch] += valid_kd_loss

        # save the validation accuracies
        if valid_taw_accuracie is not None:
            self.valid_taw_accuracies[epoch] += valid_taw_accuracie
        if valid_tag_accuracies is not None:
            self.valid_tag_accuracies[epoch] += valid_tag_accuracies

    def _reset_save(self, task):
        if self.current_task != task:
            self.current_task = task
            self.total_loss = [0]*self.total_epochs
            self.train_loss = [0]*self.total_epochs
            self.kd_loss = [0]*self.total_epochs

            self.valid_total_loss = [0]*self.total_epochs
            self.valid_train_loss = [0]*self.total_epochs
            self.valid_kd_loss = [0]*self.total_epochs
            self.valid_taw_accuracies = [0]*self.total_epochs
            self.valid_tag_accuracies = [0]*self.total_epochs

    def _make_adaptive_distll_weight(self, epoch):
        if self.akd_approach == 'adaptive_1':
            if self.total_loss[epoch] == 0:
                return 1
            else:
                return self.kd_loss[epoch] / self.total_loss[epoch]
        
        elif self.akd_approach == 'adaptive_2':
            if epoch < 2:
                return 1
            else:
                return self.valid_kd_loss[epoch-1] / self.valid_kd_loss[epoch-2]
        
        elif self.akd_approach == 'adaptive_3':
            if epoch < 2:
                return 1
            else:
                return self.valid_train_loss[epoch-1] / self.valid_train_loss[epoch-2]
        
        elif self.akd_approach == 'adaptive_4':
            if epoch < 2:
                return 1
            else:
                return self.valid_total_loss[epoch-1] / self.valid_total_loss[epoch-2]
        
        elif self.akd_approach == 'adaptive_5':
            if epoch < 1:
                return 1
            else:
                return 1 - self.valid_taw_accuracies[epoch-1] / ( self.valid_tag_accuracies[epoch-1] + self.valid_taw_accuracies[epoch-1] )

    
