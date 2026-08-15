"""实验配置"""
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class ModelConfig:
    """模型配置 - 使用configs_v3最佳参数"""
    name: str
    model_type: str
    use_mim: bool = False
    use_fmg: bool = False       # 特征级缺失门控（替代/增强 MIM）
    use_missing_branch: bool = False  # 缺失感知分支（把缺失掩码作为独立信息）
    use_mask_only: bool = False       # 仅使用缺失掩码预测（验证缺失是否携带信息）
    use_group_aware: bool = False     # 是否使用特征分组嵌入
    use_gnn: bool = False             # 是否使用特征图神经网络
    use_graphmim: bool = False        # 是否使用 GraphMIM（GNN + 组级池化）
    group_ids: list = None            # 每个特征所属的组 ID
    group_emb_dim: int = 4            # 组嵌入维度
    gnn_hidden: int = 32              # GNN 隐藏维度
    gnn_layers: int = 2               # GNN 层数
    gnn_num_heads: int = 1            # GNN 注意力头数
    use_physics_loss: bool = False    # 是否启用物理约束损失
    physics_loss_types: list = None   # 物理损失类型，如 ['monotonicity', 'smoothness']
    physics_loss_weights: dict = None # 物理损失权重
    use_curriculum: bool = False # 多缺失率课程学习
    fmg_hidden_dim: int = None  # FMG 门控隐藏层维度（None 表示默认）
    fmg_n_layers: int = 2       # FMG 门控 MLP 层数
    # 训练策略：
    #   'baseline'            - 仅在完整数据上训练
    #   'multi_missing_rate'  - 在多个缺失率混合数据上训练，但不拼接缺失指示器
    #   'mim'                 - 在多个缺失率混合数据上训练，并拼接缺失指示器
    strategy: str = 'baseline'
    # MLP参数
    hidden_layers: List[int] = None
    # LSTM/GRU参数
    hidden_size: int = 64
    num_layers: int = 2  # configs_v3最佳: 2层
    seq_len: int = 5
    # CNN参数
    channels: List[int] = None
    kernel_size: int = 3
    # Transformer参数
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1  # 添加dropout参数
    # Neural CDE / GRIN 等模型参数
    hidden_dim: int = 64  # 通用隐藏维度（用于 Neural CDE、GRIN 等）


@dataclass
class ExperimentConfig:
    """实验配置"""
    # 实验基本信息
    name: str = "soh_mim_xjtu"
    timestamp: str = None
    random_seed: int = 42
    
    # 重复实验次数
    n_repeats: int = 100
    
    # 缺失率设置
    missing_rates: List[float] = field(default_factory=lambda: [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    ])
    training_missing_rates: List[float] = field(default_factory=lambda: [
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    ])
    missing_pattern: str = 'bernoulli'  # bernoulli, block, channel, state_dependent, mixed
    
    # PINN 权重覆盖（用于敏感性分析）
    physics_mono: float = None
    physics_smooth: float = None
    
    # 数据设置
    dataset_name: str = "XJTU"  # XJTU, HUST, MIT, TJU, NASA
    data_dir: str = "./data/XJTU data"
    batch: str = "3C"  # XJTU: 2C,3C,R2.5,R3,RW,Sim_satellite; HUST: 1-10; MIT: 日期; TJU: Dataset_X_...; NASA: all
    test_size: float = 0.25  # 按电池划分
    val_size: float = 0.25
    
    # 特征列（16个）
    feature_cols: List[str] = field(default_factory=lambda: [
        'voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
        'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
        'current mean', 'current std', 'current kurtosis', 'current skewness',
        'CV Q', 'CV charge time', 'current slope', 'current entropy'
    ])
    target_col: str = 'capacity'
    seq_len: int = 5
    
    # 训练设置
    epochs: int = 100
    batch_size: int = 32
    lr: float = 0.001
    optimizer: str = "Adam"
    weight_decay: float = 0.0
    early_stopping_patience: int = 15
    
    # 路径设置
    results_dir: str = "./results"
    
    # 模型过滤：'all' 或模型名称/类型列表
    # 例如: ['mlp'] 会匹配 MLP, MLP-MIM, MLP-MultiMR
    # 例如: ['mlp-mim'] 精确匹配 MLP-MIM
    # 例如: ['fmg'] 匹配所有 FMG 配置
    model_filter: List[str] = field(default_factory=lambda: ['all'])
    
    # 是否在 get_model_configs 中包含 FMG 相关配置
    include_fmg: bool = False
    
    # 模型配置 - configs_v3最佳参数 (已移除XGBoost)
    def get_model_configs(self) -> List[ModelConfig]:
        """获取所有模型配置 - 使用configs_v3架构搜索最佳参数
        
        根据 self.model_filter 过滤返回的配置。
        """
        # 根据特征名推断分组：含 current 或 CV 归为电流组(1)，其余为电压/充放电组(0)
        group_ids = [
            1 if ('current' in col.lower() or 'cv ' in col.lower()) else 0
            for col in self.feature_cols
        ]

        configs = [
            # MLP: [192,96,48,24] 4层, dropout=0.15, MAE=0.0128
            ModelConfig(
                name='MLP', 
                model_type='mlp',
                hidden_layers=[192, 96, 48, 24],
                dropout=0.15,
                use_mim=False
            ),
            ModelConfig(
                name='MLP-MIM', 
                model_type='mlp',
                hidden_layers=[192, 96, 48, 24],
                dropout=0.15,
                use_mim=True,
                strategy='mim'
            ),
            # LSTM: h=48, l=2, dropout=0.2, MAE=0.0077
            ModelConfig(
                name='LSTM',
                model_type='lstm',
                hidden_size=48,
                num_layers=2,
                dropout=0.2,
                seq_len=self.seq_len,
                use_mim=False
            ),
            ModelConfig(
                name='LSTM-MIM',
                model_type='lstm',
                hidden_size=48,
                num_layers=2,
                dropout=0.2,
                seq_len=self.seq_len,
                use_mim=True,
                strategy='mim'
            ),
            # GRU: h=64, l=2, dropout=0.2, MAE=0.0069
            ModelConfig(
                name='GRU',
                model_type='gru',
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
                seq_len=self.seq_len,
                use_mim=False
            ),
            ModelConfig(
                name='GRU-MIM',
                model_type='gru',
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
                seq_len=self.seq_len,
                use_mim=True,
                strategy='mim'
            ),
            # CNN1D: [72,32], kernel=4, dropout=0.1, MAE=0.0057 (最佳)
            ModelConfig(
                name='CNN1D',
                model_type='cnn1d',
                channels=[72, 32],
                kernel_size=4,
                dropout=0.1,
                seq_len=self.seq_len,
                use_mim=False
            ),
            ModelConfig(
                name='CNN1D-MIM',
                model_type='cnn1d',
                channels=[72, 32],
                kernel_size=4,
                dropout=0.1,
                seq_len=self.seq_len,
                use_mim=True,
                strategy='mim'
            ),
            # === Reviewer #1 要求的基线：多缺失率训练但不使用缺失指示器 ===
            ModelConfig(
                name='MLP-MultiMR',
                model_type='mlp',
                hidden_layers=[192, 96, 48, 24],
                dropout=0.15,
                use_mim=False,
                strategy='multi_missing_rate'
            ),
            ModelConfig(
                name='LSTM-MultiMR',
                model_type='lstm',
                hidden_size=48,
                num_layers=2,
                dropout=0.2,
                seq_len=self.seq_len,
                use_mim=False,
                strategy='multi_missing_rate'
            ),
            ModelConfig(
                name='GRU-MultiMR',
                model_type='gru',
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
                seq_len=self.seq_len,
                use_mim=False,
                strategy='multi_missing_rate'
            ),
            ModelConfig(
                name='CNN1D-MultiMR',
                model_type='cnn1d',
                channels=[72, 32],
                kernel_size=4,
                dropout=0.1,
                seq_len=self.seq_len,
                use_mim=False,
                strategy='multi_missing_rate'
            ),
            # === 公平消融：在所有缺失率上均匀混合训练 ===
            ModelConfig(
                name='MLP-MultiMR-Uniform',
                model_type='mlp',
                hidden_layers=[192, 96, 48, 24],
                dropout=0.15,
                use_mim=False,
                strategy='uniform_multi_rate'
            ),
            ModelConfig(
                name='MLP-MIM-Uniform',
                model_type='mlp',
                hidden_layers=[192, 96, 48, 24],
                dropout=0.15,
                use_mim=True,
                strategy='uniform_multi_rate'
            ),
            ModelConfig(
                name='LSTM-MultiMR-Uniform',
                model_type='lstm',
                hidden_size=48,
                num_layers=2,
                dropout=0.2,
                seq_len=self.seq_len,
                use_mim=False,
                strategy='uniform_multi_rate'
            ),
            ModelConfig(
                name='LSTM-MIM-Uniform',
                model_type='lstm',
                hidden_size=48,
                num_layers=2,
                dropout=0.2,
                seq_len=self.seq_len,
                use_mim=True,
                strategy='uniform_multi_rate'
            ),
            ModelConfig(
                name='GRU-MultiMR-Uniform',
                model_type='gru',
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
                seq_len=self.seq_len,
                use_mim=False,
                strategy='uniform_multi_rate'
            ),
            ModelConfig(
                name='CNN1D-MultiMR-Uniform',
                model_type='cnn1d',
                channels=[72, 32],
                kernel_size=4,
                dropout=0.1,
                seq_len=self.seq_len,
                use_mim=False,
                strategy='uniform_multi_rate'
            ),
            # === Transformer 强时序基线 ===
            ModelConfig(
                name='Transformer-MIM-Uniform',
                model_type='transformer',
                d_model=64,
                nhead=4,
                num_layers=2,
                dim_feedforward=128,
                dropout=0.1,
                seq_len=self.seq_len,
                use_mim=True,
                strategy='uniform_multi_rate'
            ),
            ModelConfig(
                name='Transformer-MultiMR-Uniform',
                model_type='transformer',
                d_model=64,
                nhead=4,
                num_layers=2,
                dim_feedforward=128,
                dropout=0.1,
                seq_len=self.seq_len,
                use_mim=False,
                strategy='uniform_multi_rate'
            ),
            # === iTransformer：变量作为 token，适合 channel missing ===
            ModelConfig(
                name='iTransformer-MIM-Uniform',
                model_type='itransformer',
                d_model=64,
                nhead=4,
                num_layers=2,
                dim_feedforward=128,
                dropout=0.1,
                seq_len=self.seq_len,
                use_mim=True,
                strategy='uniform_multi_rate'
            ),
            ModelConfig(
                name='iTransformer-MultiMR-Uniform',
                model_type='itransformer',
                d_model=64,
                nhead=4,
                num_layers=2,
                dim_feedforward=128,
                dropout=0.1,
                seq_len=self.seq_len,
                use_mim=False,
                strategy='uniform_multi_rate'
            ),
            # === SAITS：对角掩码自注意力插补 ===
            ModelConfig(
                name='SAITS-MIM-Uniform',
                model_type='saits',
                d_model=64,
                nhead=4,
                num_layers=2,
                dim_feedforward=128,
                dropout=0.1,
                seq_len=self.seq_len,
                use_mim=True,
                strategy='uniform_multi_rate'
            ),
            ModelConfig(
                name='SAITS-MultiMR-Uniform',
                model_type='saits',
                d_model=64,
                nhead=4,
                num_layers=2,
                dim_feedforward=128,
                dropout=0.1,
                seq_len=self.seq_len,
                use_mim=False,
                strategy='uniform_multi_rate'
            ),
            # === Neural CDE：连续时间序列模型 ===
            ModelConfig(
                name='NeuralCDE-MIM-Uniform',
                model_type='neural_cde',
                hidden_dim=64,
                num_layers=2,
                dropout=0.1,
                seq_len=self.seq_len,
                use_mim=True,
                strategy='uniform_multi_rate'
            ),
            ModelConfig(
                name='NeuralCDE-MultiMR-Uniform',
                model_type='neural_cde',
                hidden_dim=64,
                num_layers=2,
                dropout=0.1,
                seq_len=self.seq_len,
                use_mim=False,
                strategy='uniform_multi_rate'
            ),
            # === GRIN：图循环插补网络 ===
            ModelConfig(
                name='GRIN-MIM-Uniform',
                model_type='grin',
                hidden_dim=64,
                num_layers=2,
                dropout=0.1,
                seq_len=self.seq_len,
                use_mim=True,
                group_ids=group_ids,
                group_emb_dim=4,
                gnn_hidden=32,
                gnn_layers=2,
                gnn_num_heads=1,
                strategy='uniform_multi_rate'
            ),
            ModelConfig(
                name='GRIN-MultiMR-Uniform',
                model_type='grin',
                hidden_dim=64,
                num_layers=2,
                dropout=0.1,
                seq_len=self.seq_len,
                use_mim=False,
                group_ids=group_ids,
                group_emb_dim=4,
                gnn_hidden=32,
                gnn_layers=2,
                gnn_num_heads=1,
                strategy='uniform_multi_rate'
            ),
            # === 组感知 MIM（利用电压/电流分组先验）===
            ModelConfig(
                name='MLP-GroupMIM-Uniform',
                model_type='mlp',
                hidden_layers=[192, 96, 48, 24],
                dropout=0.15,
                use_mim=True,
                use_group_aware=True,
                group_ids=group_ids,
                group_emb_dim=4,
                strategy='uniform_multi_rate'
            ),
            ModelConfig(
                name='LSTM-GroupMIM-Uniform',
                model_type='lstm',
                hidden_size=48,
                num_layers=2,
                dropout=0.2,
                seq_len=self.seq_len,
                use_mim=True,
                use_group_aware=True,
                group_ids=group_ids,
                group_emb_dim=4,
                strategy='uniform_multi_rate'
            ),
            # === 特征图神经网络缺失感知 ===
            ModelConfig(
                name='MLP-GNN-Uniform',
                model_type='mlp',
                hidden_layers=[192, 96, 48, 24],
                dropout=0.15,
                use_mim=True,
                use_gnn=True,
                group_ids=group_ids,
                group_emb_dim=4,
                gnn_hidden=32,
                gnn_layers=2,
                strategy='uniform_multi_rate'
            ),
            ModelConfig(
                name='LSTM-GNN-Uniform',
                model_type='lstm',
                hidden_size=48,
                num_layers=2,
                dropout=0.2,
                seq_len=self.seq_len,
                use_mim=True,
                use_gnn=True,
                group_ids=group_ids,
                group_emb_dim=4,
                gnn_hidden=32,
                gnn_layers=2,
                strategy='uniform_multi_rate'
            ),
        ]

        # === 新增：GraphMIM（GNN + 组级池化）配置 ===
        # 默认不加入 all，避免 --models all 时实验量过大；
        # 通过 model_filter=['graphmim'] 或显式名称选择。
        configs += [
            ModelConfig(
                name='MLP-GraphMIM-Uniform',
                model_type='mlp',
                hidden_layers=[192, 96, 48, 24],
                dropout=0.15,
                use_graphmim=True,
                use_mim=True,
                group_ids=group_ids,
                group_emb_dim=4,
                gnn_hidden=32,
                gnn_layers=2,
                strategy='uniform_multi_rate'
            ),
            ModelConfig(
                name='LSTM-GraphMIM-Uniform',
                model_type='lstm',
                hidden_size=48,
                num_layers=2,
                dropout=0.2,
                seq_len=self.seq_len,
                use_graphmim=True,
                use_mim=True,
                group_ids=group_ids,
                group_emb_dim=4,
                gnn_hidden=32,
                gnn_layers=2,
                strategy='uniform_multi_rate'
            ),
            # === GraphMIM + 物理约束（PINN）===
            ModelConfig(
                name='MLP-GraphMIM-Uniform-Phys',
                model_type='mlp',
                hidden_layers=[192, 96, 48, 24],
                dropout=0.15,
                use_graphmim=True,
                use_mim=True,
                group_ids=group_ids,
                group_emb_dim=4,
                gnn_hidden=32,
                gnn_layers=2,
                strategy='uniform_multi_rate',
                use_physics_loss=True,
                physics_loss_types=['monotonicity', 'smoothness'],
                physics_loss_weights={
                    'monotonicity': self.physics_mono if self.physics_mono is not None else 0.01,
                    'smoothness': self.physics_smooth if self.physics_smooth is not None else 0.001
                }
            ),
            ModelConfig(
                name='MLP-GNN-Uniform-Phys',
                model_type='mlp',
                hidden_layers=[192, 96, 48, 24],
                dropout=0.15,
                use_gnn=True,
                use_mim=True,
                group_ids=group_ids,
                group_emb_dim=4,
                gnn_hidden=32,
                gnn_layers=2,
                strategy='uniform_multi_rate',
                use_physics_loss=True,
                physics_loss_types=['monotonicity', 'smoothness'],
                physics_loss_weights={
                    'monotonicity': self.physics_mono if self.physics_mono is not None else 0.01,
                    'smoothness': self.physics_smooth if self.physics_smooth is not None else 0.001
                }
            ),
            ModelConfig(
                name='LSTM-MIM-Uniform-Phys',
                model_type='lstm',
                hidden_size=48,
                num_layers=2,
                dropout=0.2,
                seq_len=self.seq_len,
                use_mim=True,
                strategy='uniform_multi_rate',
                use_physics_loss=True,
                physics_loss_types=['monotonicity', 'smoothness'],
                physics_loss_weights={
                    'monotonicity': self.physics_mono if self.physics_mono is not None else 0.01,
                    'smoothness': self.physics_smooth if self.physics_smooth is not None else 0.001
                }
            ),
        ]
        
        # === 新增：FMG（Feature-wise Missing Gate）配置 ===
        # 默认不加入 all，避免 --models all 时实验量爆炸；
        # 通过 model_filter=['fmg'] 或显式名称选择。
        if self.include_fmg:
            fmg_configs = [
                # FMG 本身（多缺失率 + 门控）
                ModelConfig(
                    name='MLP-FMG',
                    model_type='mlp',
                    hidden_layers=[192, 96, 48, 24],
                    dropout=0.15,
                    use_fmg=True,
                    strategy='mim'
                ),
                ModelConfig(
                    name='LSTM-FMG',
                    model_type='lstm',
                    hidden_size=48,
                    num_layers=2,
                    dropout=0.2,
                    seq_len=self.seq_len,
                    use_fmg=True,
                    strategy='mim'
                ),
                ModelConfig(
                    name='GRU-FMG',
                    model_type='gru',
                    hidden_size=64,
                    num_layers=2,
                    dropout=0.2,
                    seq_len=self.seq_len,
                    use_fmg=True,
                    strategy='mim'
                ),
                ModelConfig(
                    name='CNN1D-FMG',
                    model_type='cnn1d',
                    channels=[72, 32],
                    kernel_size=4,
                    dropout=0.1,
                    seq_len=self.seq_len,
                    use_fmg=True,
                    strategy='mim'
                ),
                # FMG + uniform multi-rate（公平对比 MultiMR-Uniform）
                ModelConfig(
                    name='MLP-FMG-Uniform',
                    model_type='mlp',
                    hidden_layers=[192, 96, 48, 24],
                    dropout=0.15,
                    use_fmg=True,
                    strategy='uniform_multi_rate'
                ),
                ModelConfig(
                    name='LSTM-FMG-Uniform',
                    model_type='lstm',
                    hidden_size=48,
                    num_layers=2,
                    dropout=0.2,
                    seq_len=self.seq_len,
                    use_fmg=True,
                    strategy='uniform_multi_rate'
                ),
                ModelConfig(
                    name='GRU-FMG-Uniform',
                    model_type='gru',
                    hidden_size=64,
                    num_layers=2,
                    dropout=0.2,
                    seq_len=self.seq_len,
                    use_fmg=True,
                    strategy='uniform_multi_rate'
                ),
                ModelConfig(
                    name='CNN1D-FMG-Uniform',
                    model_type='cnn1d',
                    channels=[72, 32],
                    kernel_size=4,
                    dropout=0.1,
                    seq_len=self.seq_len,
                    use_fmg=True,
                    strategy='uniform_multi_rate'
                ),
                # FMG 门控容量消融（基于 LSTM + Uniform）
                ModelConfig(
                    name='LSTM-FMG-Hid8',
                    model_type='lstm',
                    hidden_size=48,
                    num_layers=2,
                    dropout=0.2,
                    seq_len=self.seq_len,
                    use_fmg=True,
                    fmg_hidden_dim=8,
                    strategy='uniform_multi_rate'
                ),
                ModelConfig(
                    name='LSTM-FMG-Hid32',
                    model_type='lstm',
                    hidden_size=48,
                    num_layers=2,
                    dropout=0.2,
                    seq_len=self.seq_len,
                    use_fmg=True,
                    fmg_hidden_dim=32,
                    strategy='uniform_multi_rate'
                ),
                ModelConfig(
                    name='LSTM-FMG-Hid64',
                    model_type='lstm',
                    hidden_size=48,
                    num_layers=2,
                    dropout=0.2,
                    seq_len=self.seq_len,
                    use_fmg=True,
                    fmg_hidden_dim=64,
                    strategy='uniform_multi_rate'
                ),
                ModelConfig(
                    name='LSTM-FMG-Linear',
                    model_type='lstm',
                    hidden_size=48,
                    num_layers=2,
                    dropout=0.2,
                    seq_len=self.seq_len,
                    use_fmg=True,
                    fmg_n_layers=1,
                    strategy='uniform_multi_rate'
                ),
                ModelConfig(
                    name='LSTM-FMG-Deep3',
                    model_type='lstm',
                    hidden_size=48,
                    num_layers=2,
                    dropout=0.2,
                    seq_len=self.seq_len,
                    use_fmg=True,
                    fmg_n_layers=3,
                    strategy='uniform_multi_rate'
                ),
                # FMG + 课程学习
                ModelConfig(
                    name='MLP-FMG-CMR',
                    model_type='mlp',
                    hidden_layers=[192, 96, 48, 24],
                    dropout=0.15,
                    use_fmg=True,
                    use_curriculum=True,
                    strategy='mim'
                ),
                ModelConfig(
                    name='LSTM-FMG-CMR',
                    model_type='lstm',
                    hidden_size=48,
                    num_layers=2,
                    dropout=0.2,
                    seq_len=self.seq_len,
                    use_fmg=True,
                    use_curriculum=True,
                    strategy='mim'
                ),
                # === 缺失感知分支（Missingness as information）===
                ModelConfig(
                    name='MLP-MissingAware-Uniform',
                    model_type='mlp',
                    hidden_layers=[192, 96, 48, 24],
                    dropout=0.15,
                    use_mim=True,
                    use_missing_branch=True,
                    strategy='uniform_multi_rate'
                ),
                ModelConfig(
                    name='LSTM-MissingAware-Uniform',
                    model_type='lstm',
                    hidden_size=48,
                    num_layers=2,
                    dropout=0.2,
                    seq_len=self.seq_len,
                    use_mim=True,
                    use_missing_branch=True,
                    strategy='uniform_multi_rate'
                ),
                ModelConfig(
                    name='GRU-MissingAware-Uniform',
                    model_type='gru',
                    hidden_size=64,
                    num_layers=2,
                    dropout=0.2,
                    seq_len=self.seq_len,
                    use_mim=True,
                    use_missing_branch=True,
                    strategy='uniform_multi_rate'
                ),
                # 只用缺失掩码的基线（验证缺失本身是否携带 SOH 信息）
                ModelConfig(
                    name='MLP-MaskOnly-Uniform',
                    model_type='mlp',
                    hidden_layers=[192, 96, 48, 24],
                    dropout=0.15,
                    use_mim=True,
                    use_mask_only=True,
                    strategy='uniform_multi_rate'
                ),
                ModelConfig(
                    name='LSTM-MaskOnly-Uniform',
                    model_type='lstm',
                    hidden_size=48,
                    num_layers=2,
                    dropout=0.2,
                    seq_len=self.seq_len,
                    use_mim=True,
                    use_mask_only=True,
                    strategy='uniform_multi_rate'
                ),
                ModelConfig(
                    name='GRU-FMG-CMR',
                    model_type='gru',
                    hidden_size=64,
                    num_layers=2,
                    dropout=0.2,
                    seq_len=self.seq_len,
                    use_fmg=True,
                    use_curriculum=True,
                    strategy='mim'
                ),
                ModelConfig(
                    name='CNN1D-FMG-CMR',
                    model_type='cnn1d',
                    channels=[72, 32],
                    kernel_size=4,
                    dropout=0.1,
                    seq_len=self.seq_len,
                    use_fmg=True,
                    use_curriculum=True,
                    strategy='mim'
                ),
            ]
            configs.extend(fmg_configs)
        
        # 应用模型过滤
        if not self.model_filter or 'all' in [m.lower() for m in self.model_filter]:
            return configs
        
        selected = []
        filter_set = {m.lower() for m in self.model_filter}
        for cfg in configs:
            if cfg.name.lower() in filter_set or cfg.model_type.lower() in filter_set:
                selected.append(cfg)
        
        return selected if selected else configs
