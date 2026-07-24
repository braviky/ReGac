import numpy as np
import torch
from easydict import EasyDict

from .misc import BlackHole


def get_optimizer(cfg, model):
    if cfg.type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2,)
        )
    else:
        raise NotImplementedError('Optimizer not supported: %s' % cfg.type)


def get_scheduler(cfg, optimizer):
    if cfg.type is None:
        return BlackHole()
    elif cfg.type == 'plateau':
        return DeepSpeedPlateauScheduler(
            optimizer,
            factor=cfg.factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr,
        )
    elif cfg.type == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.milestones,
            gamma=cfg.gamma,
        )
    elif cfg.type == 'exp':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg.gamma,
        )
    elif cfg.type is None:
        return BlackHole()
    else:
        raise NotImplementedError('Scheduler not supported: %s' % cfg.type)


class DeepSpeedPlateauScheduler:
    """
    兼容 DeepSpeed optimizer 的 ReduceLROnPlateau 替代方案
    直接操作 optimizer.param_groups['lr']，绕过 PyTorch scheduler 的类型检查
    """
    def __init__(self, optimizer, factor=0.1, patience=10, min_lr=1e-6):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.num_bad_epochs = 0

    def step(self, metric=None):
        """
        Step scheduler. If metric is provided (validation loss), use it for plateau detection.
        If metric is None, do nothing (for compatibility with schedulers that don't need metric).
        """
        if metric is None:
            return

        if metric < self.best_loss:
            self.best_loss = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0

    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr

    def get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


def get_warmup_sched(cfg, optimizer):
    if cfg is None: return BlackHole()
    lambdas = [lambda it: (it / cfg.max_iters) if it <= cfg.max_iters else 1 for _ in optimizer.param_groups]
    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
    return warmup_sched


def log_losses(out, it, tag, logger=BlackHole(), writer=BlackHole(), others={}, train_weights=None, lr=None, loss_dict_raw=None, weight_info=None):
    """
    记录损失到日志和 TensorBoard

    Args:
        out: 损失字典（加权后）
        it: 当前迭代数
        tag: 日志标签
        logger: 日志记录器
        writer: TensorBoard writer
        others: 其他指标
        train_weights: 参与反向传播的权重字典，用于区分是否参与训练
        lr: 学习率
        loss_dict_raw: 原始损失字典（未加权）
        weight_info: 权重信息字典
    """
    key_mapping = {
        'contact': 'interface',
        'clash': 'interface',
        'bsa': 'interface',
        'shape': 'interface',
    }

    logstr = '[%s] Iter %05d' % (tag, it)
    logstr += ' | loss %.4f' % out['overall'].item()

    trainable_losses = {}
    non_trainable_losses = {}

    for k, v in out.items():
        if k == 'overall': continue
        if not isinstance(v, torch.Tensor): continue
        weight_key = key_mapping.get(k, k)
        raw_loss = loss_dict_raw.get(k+"_mean", loss_dict_raw.get(k)) if loss_dict_raw else None
        has_gradient = raw_loss is not None and isinstance(raw_loss, torch.Tensor) and raw_loss.requires_grad
        if not has_gradient:
            non_trainable_losses[k] = v.item()
            raw_val = loss_dict_raw.get(k+'_mean', loss_dict_raw.get(k)) if loss_dict_raw else None
            weight_val = weight_info.get(k) if weight_info else None
            if raw_val is not None and weight_val is not None:
                raw_str = raw_val.item() if isinstance(raw_val, torch.Tensor) else raw_val
                weight_str = weight_val.item() if isinstance(weight_val, torch.Tensor) else weight_val
                logstr += ' | loss_nograd(%s) %.4f (w=%.2f, raw=%.4f)' % (k, v.item(), weight_str, raw_str)
            else:
                logstr += ' | loss_nograd(%s) %.4f' % (k, v.item())
        else:
            trainable_losses[k] = v.item()
            raw_val = loss_dict_raw.get(k+'_mean', loss_dict_raw.get(k)) if loss_dict_raw else None
            weight_val = weight_info.get(k) if weight_info else None
            if raw_val is not None and weight_val is not None:
                raw_str = raw_val.item() if isinstance(raw_val, torch.Tensor) else raw_val
                weight_str = weight_val.item() if isinstance(weight_val, torch.Tensor) else weight_val
                logstr += ' | loss(%s) %.4f (w=%.2f, raw=%.4f)' % (k, v.item(), weight_str, raw_str)
            else:
                logstr += ' | loss(%s) %.4f' % (k, v.item())

    if lr is not None:
        logstr += ' | lr %.6f' % lr
        writer.add_scalar('%s/lr' % tag, lr, it)

    if loss_dict_raw is not None:
        for mk, mv in sorted(loss_dict_raw.items()):
            if not mk.startswith('monitor_'):
                continue
            if isinstance(mv, torch.Tensor):
                mv = mv.mean().item() if mv.numel() > 1 else mv.item()
            if mv is None:
                continue
            logstr += ' | %s %.4f' % (mk, mv)
            writer.add_scalar('%s/%s' % (tag, mk), mv, it)

    for k, v in others.items():
        if v is None:
            continue
        if isinstance(v, str):
            logstr += ' | %s %s' % (k, v)
        else:
            logstr += ' | %s %.4f' % (k, v)
            writer.add_scalar('%s/%s' % (tag, k), v, it)

    logger.info(logstr)

    if weight_info is not None and it % 100 == 0:
        core_keys = ['cdr_rot', 'cdr_pos', 'cdr_seq', 'epitope_rot', 'epitope_pos']
        weight_parts = []
        for k in core_keys:
            if k in weight_info:
                w = weight_info[k]
                if isinstance(w, torch.Tensor):
                    w = w.mean().item() if w.numel() > 1 else w.item()
                weight_parts.append('%s=%.2f' % (k.replace('cdr_', 'c_').replace('epitope_', 'e_'), w))
        if weight_parts:
            logger.info('[%s] Weights: %s' % (tag, ' | '.join(weight_parts)))

        try:
            from diffab_251166.modules.common.bfactor_utils import debug_log, DEBUG_WEIGHTS
            if DEBUG_WEIGHTS:
                debug_log(f"[Iter {it}] Weights: {' | '.join(weight_parts)}")
        except:
            pass

        geom_keys = ['geom_bone_scale', 'geom_omega_scale', 'geom_alpha_b', 'geom_alpha_o']
        geom_parts = []
        for k in geom_keys:
            if k in weight_info:
                w = weight_info[k]
                geom_parts.append('%s=%.4f' % (k.replace('geom_', ''), w))
        if geom_parts:
            logger.info('[%s] GeomLossScaler: %s' % (tag, ' | '.join(geom_parts)))

        for k, w in weight_info.items():
            if isinstance(w, torch.Tensor):
                w = w.mean().item() if w.numel() > 1 else w.item()
            writer.add_scalar('%s/weight/%s' % (tag, k), w, it)

    for k, v in out.items():
        if not isinstance(v, torch.Tensor): continue
        v_scalar = v.item()  # 转换成 Python scalar
        if k == 'overall':
            writer.add_scalar('%s/loss' % tag, v_scalar, it)
        else:
            weight_key = key_mapping.get(k, k)
            if not v.requires_grad:
                writer.add_scalar('%s/no_grad/loss_%s' % (tag, k), v_scalar, it)
            else:
                writer.add_scalar('%s/loss_%s' % (tag, k), v_scalar, it)

    if loss_dict_raw is not None:
        for k, v in loss_dict_raw.items():
            if not isinstance(v, torch.Tensor): continue
            v_scalar = v.mean().item() if v.numel() > 1 else v.item()
            if k == 'overall':
                writer.add_scalar('%s/loss_raw' % tag, v_scalar, it)
            else:
                writer.add_scalar('%s/loss_raw_%s' % (tag, k), v_scalar, it)

    writer.flush()

    return trainable_losses, non_trainable_losses


class ValidationLossTape(object):

    def __init__(self):
        super().__init__()
        self.accumulate = {}
        self.accumulate_raw = {}  # 存储原始（未加权）损失
        self.others = {}
        self.total = 0
        self.avg = None
        self.avg_raw = None
        self.avg_others = None

    def update(self, out, n, others={}, out_raw=None):
        self.total += n
        for k, v in out.items():
            if not isinstance(v, torch.Tensor): continue  # 跳过非 tensor
            if k not in self.accumulate:
                self.accumulate[k] = v.clone().detach()
            else:
                self.accumulate[k] += v.clone().detach()

        if out_raw is not None:
            for k, v in out_raw.items():
                if not isinstance(v, torch.Tensor): continue
                v_scalar = v.mean() if v.numel() > 1 else v
                if k not in self.accumulate_raw:
                    self.accumulate_raw[k] = v_scalar.clone().detach()
                else:
                    self.accumulate_raw[k] += v_scalar.clone().detach()

        for k, v in others.items():
            if not isinstance(v, torch.Tensor): continue  # 跳过非 tensor
            if k not in self.others:
                self.others[k] = v.clone().detach()
            else:
                self.others[k] += v.clone().detach()

    def compute_avg(self):
        avg = EasyDict({k: v / self.total for k, v in self.accumulate.items()})
        avg_raw = EasyDict({k: v / self.total for k, v in self.accumulate_raw.items()}) if self.accumulate_raw else None
        avg_others = EasyDict({k: v / self.total for k, v in self.others.items()})
        self.avg = avg
        self.avg_raw = avg_raw
        self.avg_others = avg_others
        return avg, avg_others

    def log(self, it, logger=BlackHole(), writer=BlackHole(), tag='val', train_weights=None, lr=None, weight_info=None):
        if self.avg is None or self.avg_others is None:
            self.avg, self.avg_others = self.compute_avg()
        log_losses(self.avg, it, tag, logger, writer, others=self.avg_others, train_weights=train_weights, lr=lr, loss_dict_raw=self.avg_raw, weight_info=weight_info)
        return self.avg['overall']


def recursive_to(obj, device, dtype=None):
    if isinstance(obj, torch.Tensor):
        if device == 'cpu':
            tensor = obj.cpu()
        else:
            try:
                tensor = obj.cuda(device=device, non_blocking=True)
            except RuntimeError:
                tensor = obj.to(device)

        if dtype is not None and tensor.is_floating_point():
            return tensor.to(dtype)

        return tensor

    elif isinstance(obj, list):
        return [recursive_to(o, device=device, dtype=dtype) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, device=device, dtype=dtype) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device=device, dtype=dtype) for k, v in obj.items()}

    else:
        return obj


def reweight_loss_by_sequence_length(length, max_length, mode='sqrt'):
    if mode == 'sqrt':
        w = np.sqrt(length / max_length)
    elif mode == 'linear':
        w = length / max_length
    elif mode is None:
        w = 1.0
    else:
        raise ValueError('Unknown reweighting mode: %s' % mode)
    return w


def sum_weighted_losses(losses, weights):
    """
    Args:
        losses:     Dict of scalar tensors.
        weights:    Dict of weights.
    """
    if weights is None:
        return sum(losses.values())

    key_mapping = {
        'contact': 'interface',
        'clash': 'interface',
    }

    first_tensor = None
    for v in losses.values():
        if isinstance(v, torch.Tensor):
            first_tensor = v
            break

    if first_tensor is None:
        return torch.tensor(0.0, requires_grad=True)

    loss = None
    for k, v in losses.items():
        if k in weights:
            w = weights[k]
        elif k in key_mapping and key_mapping[k] in weights:
            w = weights[key_mapping[k]]
        else:
            continue

        if isinstance(v, torch.Tensor):
            if loss is None:
                loss = w * v
            else:
                loss = loss + w * v

    if loss is None:
        return torch.tensor(0.0, device=first_tensor.device, requires_grad=True)

    return loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
