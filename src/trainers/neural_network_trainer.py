"""神经网络训练器"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class NeuralNetworkTrainer:
    """神经网络训练器"""
    
    def __init__(self, model, device: str = 'cpu'):
        """
        Args:
            model: PyTorch模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss()
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None,
              epochs: int = 100, lr: float = 0.001, 
              weight_decay: float = 0.0, patience: int = 15) -> dict:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            lr: 学习率
            weight_decay: 权重衰减
            patience: 早停耐心值
            
        Returns:
            dict: 训练历史
        """
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                # 确保输出和目标维度一致
                outputs = outputs.view(-1)
                batch_y = batch_y.view(-1)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(batch_x)
            
            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)
            
            # 验证阶段
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_x)
                        # 确保输出和目标维度一致
                        outputs = outputs.view(-1)
                        batch_y = batch_y.view(-1)
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item() * len(batch_x)
                
                val_loss /= len(val_loader.dataset)
                history['val_loss'].append(val_loss)
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型状态
                    best_model_state = {
                        k: v.cpu().clone() 
                        for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"早停于 epoch {epoch+1}")
                    break
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f", Val Loss: {val_loss:.6f}"
                print(msg)
        
        training_time = time.time() - start_time
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        history['training_time'] = training_time
        history['best_epoch'] = len(history['train_loss']) - patience_counter
        
        return history
