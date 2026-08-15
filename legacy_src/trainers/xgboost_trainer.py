"""XGBoost训练器"""
import time
import numpy as np
from sklearn.metrics import mean_squared_error


class XGBoostTrainer:
    """XGBoost训练器"""
    
    def __init__(self, model):
        """
        Args:
            model: XGBoost模型
        """
        self.model = model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> dict:
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            
        Returns:
            dict: 训练历史
        """
        start_time = time.time()
        
        # 训练
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        training_time = time.time() - start_time
        
        # 计算训练集损失
        train_pred = self.model.predict(X_train)
        train_loss = mean_squared_error(y_train, train_pred)
        
        history = {
            'train_loss': [train_loss],
            'training_time': training_time
        }
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_loss = mean_squared_error(y_val, val_pred)
            history['val_loss'] = [val_loss]
        
        return history
