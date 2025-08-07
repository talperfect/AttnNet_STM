import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from matplotlib import ticker
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------
plt.style.use('classic')
plt.rcParams['font.size'] = 22
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['figure.facecolor'] = '1'
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 2))
# -----------------------------------------------------------------------

# 定义多头注意力模型
class MultiHeadAttentionANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
        super(MultiHeadAttentionANN, self).__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)  # 等价于 Dense(hidden_dim)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.output_layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        x_proj = self.input_proj(x)  # (batch_size, 1, hidden_dim)

        # Multi-head self-attention: query = key = value = x_proj
        attn_output, _ = self.mha(x_proj, x_proj, x_proj)  # (batch_size, 1, hidden_dim)
        x = self.norm1(attn_output + x_proj)  # 残差连接 + LayerNorm

        # Feed-forward network
        ffn_output = self.ffn(x)  # (batch_size, 1, hidden_dim)
        x = self.norm2(x + ffn_output)  # 残差连接 + LayerNorm

        x = x.squeeze(1)  # 去掉序列维度 -> (batch_size, hidden_dim)
        output = self.output_layer2(x)
        return output

if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dfx = pd.read_csv('MAT_data&res\\X_train.csv', header=None)
    dfy = pd.read_csv('MAT_data&res\\y_train.csv', header=None)

    x_train = torch.tensor(dfx.to_numpy(), dtype=torch.float32).to(device)
    y_train = torch.tensor(dfy.to_numpy(), dtype=torch.float32).to(device)

    # 数据准备
    # 假设有1000个样本，每个样本有10个特征
    input_dim = 84
    output_dim = 1
    hidden_dim = 96
    num_heads = 12
    batch_size = 32


    # 创建数据加载器
    dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = MultiHeadAttentionANN(input_dim, hidden_dim, output_dim, num_heads).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000636)
    MSE_zu=[]

    # 训练模型
    num_epochs = 582
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        total_relative_error = []
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算相对误差

            relative_error = (torch.abs(outputs - targets) / torch.abs(targets)).mean().item()
            total_relative_error.append(relative_error)

        # 每个epoch输出MSE
        avg_loss = running_loss / len(train_loader)
        MSE_zu.append([avg_loss,np.mean(total_relative_error)])
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, relative_error: {np.mean(total_relative_error):.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'MAT_data&res\\multihead_attention_ann.pth')
    np.savetxt('MAT_data&res\\output.csv', np.array(MSE_zu), delimiter=',', fmt='%.4f')

#最佳超参数: {'hidden_size': 96, 'num_heads': 12, 'lr': 0.0006364389541549915, 'epochs': 582}, 最佳目标值: 0.045059779032814405