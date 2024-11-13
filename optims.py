import math

class LinearWarmupCosineLRScheduler:
    def __init__(
        self,
        optimizer,
        min_lr_list,
        init_lr_list,
        warmup_steps=0,
        warmup_start_lr_list=None,
        **kwargs
    ):
        self.optimizer = optimizer

        self.min_lr_list = min_lr_list
        self.init_lr_list = init_lr_list
        self.warmup_steps = warmup_steps
        self.warmup_start_lr_list = warmup_start_lr_list if warmup_start_lr_list is not None else init_lr_list

    def step(self, cur_step, cur_epoch, max_step):
        for i, param_group in enumerate(self.optimizer.param_groups):
            if cur_epoch == 0 and cur_step < self.warmup_steps:
                lr = self.warmup_lr_schedule(cur_step, self.warmup_start_lr_list[i], self.init_lr_list[i])
            else:
                lr = self.cosine_lr_schedule(cur_step - self.warmup_steps, max_step - self.warmup_steps, self.init_lr_list[i], self.min_lr_list[i])
            param_group["lr"] = lr
            
    def cosine_lr_schedule(self, step, max_step, init_lr, min_lr):
        """Decay the learning rate using cosine schedule"""
        lr = (init_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * step / max_step)) + min_lr
        return lr

    def warmup_lr_schedule(self, step, init_lr, max_lr):
        """Warmup the learning rate"""
        lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(self.warmup_steps, 1))
        return lr

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        