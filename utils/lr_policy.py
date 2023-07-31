from abc import abstractmethod


class BaseLR:
    def init_base_lr_group(self, optimizer):
        self.base_lr_group = []
        for group in optimizer.param_groups:
            self.base_lr_group.append(group["lr"])

    @abstractmethod
    def get_coef(self, cur_iter):
        pass

    @abstractmethod
    def step(self, optimizer, curr_iter):
        if not hasattr(self, "base_lr_group"):
            raise ValueError(f"Please call the method `.init_base_lr_group` first.")

        curr_coef = self.get_coef(curr_iter)

        temp_lrs = []
        for group_idx in range(len(optimizer.param_groups)):
            curr_lr = self.base_lr_group[group_idx] * curr_coef
            optimizer.param_groups[group_idx]["lr"] = curr_lr
            temp_lrs.append(curr_lr)
        return temp_lrs


class PolyLR(BaseLR):
    def __init__(self, lr_power, total_iters):
        self.lr_power = lr_power
        self.total_iters = total_iters + 0.0

    def get_coef(self, cur_iter):
        return (1 - float(cur_iter) / self.total_iters) ** self.lr_power


class WarmUpPolyLR(BaseLR):
    def __init__(self, lr_power, total_iters, warmup_steps):
        self.lr_power = lr_power
        self.total_iters = total_iters
        self.warmup_steps = warmup_steps

    def get_coef(self, cur_iter):
        if cur_iter < self.warmup_steps:
            return (cur_iter + 1) / self.warmup_steps
        else:
            cur_iter -= self.warmup_steps
            return (1 - float(cur_iter + 1) / self.total_iters) ** self.lr_power


class MultiStageLR(BaseLR):
    def __init__(self, milestones, gamma):
        self._milestones = milestones
        self._gamma = gamma

    def get_coef(self, cur_iter):
        stage_idx = 0
        for i, milestone in enumerate(self._milestones, start=1):
            if cur_iter >= milestone:
                stage_idx = i
        return self._gamma ** stage_idx
