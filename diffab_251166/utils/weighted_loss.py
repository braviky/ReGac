import torch
from torch import nn



class KLAnnealingScheduler:
    """
    用于 KL 散度损失项的线性退火调度器。
    在指定的 N_warmup 步数内，将 KL 权重从 0.0 线性增加到 lambda_kl_max。
    """

    def __init__(self, lambda_kl_max=0.05, max_step=50000):
        self.lambda_kl_max = lambda_kl_max
        self.warmup_steps = int(max_step*0.25)
        self.current_step = 0

    def step(self):
        """
        在每个训练步骤后调用，更新当前的全局步数。
        """
        self.current_step += 1

    def get_lambda_kl(self) -> float:
        """
        计算并返回当前步骤的 KL 损失权重。
        """
        if self.current_step >= self.warmup_steps:
            return self.lambda_kl_max

        progress = self.current_step / self.warmup_steps
        return self.lambda_kl_max * progress


class UncertaintyWeightedLoss(torch.nn.Module):
    """
    基于 Kendall 等人的方法，利用任务的观测不确定性动态调整重建损失权重。
    应用于: MAIN_TASKS 和 REG_TASK (KL总损失)。
    L_total = sum_i (exp(-s_i) * L_i + 0.5 * s_i)
    """

    def __init__(self, task_names):
        super().__init__()
        self.task_names = task_names

        self.log_sigmas = torch.nn.ParameterDict()

        INITIAL_REG_LOG_SIGMA = 5.0

        for name in task_names:
            initial_s = 0.0

            if name == 'L_kl_total_weighted':  # 匹配训练脚本中的 REG_TASK 名称
                initial_s = INITIAL_REG_LOG_SIGMA

            self.log_sigmas[f'log_sigma_{name}'] = torch.nn.Parameter(
                torch.tensor(initial_s, requires_grad=True)
            )

        self.log_vars = torch.nn.ParameterList(list(self.log_sigmas.values()))

    def forward(self, losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """
        计算动态加权后的总损失。
        """
        device = self.log_vars[0].device if len(self.log_vars) > 0 else torch.device('cpu')
        total_loss = torch.tensor(0.0, device=device)
        current_weights = {}

        for i, name in enumerate(self.task_names):
            if name not in losses:
                continue

            s_i = self.log_vars[i]  # 从 ParameterList 中访问
            L_i = losses[name]


            weighted_L_i = torch.exp(-s_i) * L_i

            penalty_term = 0.5 * s_i

            total_loss += weighted_L_i + penalty_term

            current_weights[f'lambda_{name}'] = torch.exp(-s_i).item()
            current_weights[f'log_sigma_{name}'] = s_i.item()

        return total_loss, current_weights
