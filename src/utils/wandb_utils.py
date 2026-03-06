"""WandB 工具模块"""
from omegaconf import DictConfig, OmegaConf


def init_wandb(cfg: DictConfig):
    """
    初始化 WandB
    
    如果 cfg.wandb.mode == "disabled"，则跳过
    """
    import wandb
    
    if cfg.wandb.mode == "disabled":
        return None
    
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.experiment.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb.mode,
        tags=cfg.wandb.tags,
    )
    return wandb.run


def finish_wandb():
    """结束 WandB 会话"""
    import wandb
    if wandb.run:
        wandb.finish()
