from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skorch import NeuralNetClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
import math, os, joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    f1_score
)

def load_model_from_pkl(file_path):
    model = joblib.load(filename=file_path)
    return model


def evaluate_model(y_true, y_pred, set_name):
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average='macro')
    pre_2 = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3])
    rec = recall_score(y_true, y_pred, average='macro')
    rec_2 = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3])
    f1 = f1_score(y_true, y_pred, average='macro')
    f1_2 = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3])

    print(f'{set_name} Evaluation:')
    print(f'  Accuracy (ACC): {acc:.4f}')
    print(f'  Kappa (KAPPA): {kappa:.4f}')
    print(f'  Precision (PRE): {pre:.4f}')
    print(f'  Precision (PRE): {pre_2}')
    print(f'  Recall (REC): {rec:.4f}')
    print(f'  Recall (REC): {rec_2}')
    print(f'  F1 Macro (F1-macro): {f1:.4f}')
    print(f'  F1 Macro (F1-macro): {f1_2}')

    print('-' * 20)


# 定义多头注意力模型
class MultiHeadAttentionANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, pretrained_model_path=None):
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
        # 加载预训练权重（如果路径提供）
        if pretrained_model_path:
            self.load_pretrained_weights(pretrained_model_path)
    def load_pretrained_weights(self, model_path):
        pretrained_dict = torch.load(model_path)
        # 只加载需要的部分，避免权重不匹配
        self.load_state_dict(pretrained_dict, strict=False)
        print(f"Pretrained weights loaded from {model_path}")

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

# 第二层：自定义计算层（示例计算 y1 * 2 + 1）
class CustomComputationLayer(nn.Module):
    def forward(self, x):
        GSI = x[:, 0]
        sigma_c = x[:, 1]
        H = x[:, 2]

        GSI = GSI * 100  # 放缩

        D = 0
        mi = 6
        # sigma_c
        Ei = 10000
        ga_ma = 0.023  # 原本是0.026

        mb = mi * torch.exp((GSI - 100) / (28 - 14 * D))
        s = torch.exp((GSI - 100) / (9 - 3 * D))

        a = 0.5 + (torch.exp(-GSI / 15) - torch.exp(torch.tensor(-20 / 3, dtype=GSI.dtype))) / 6

        sigma_cm = sigma_c * ((mb + 4 * s - a * (mb - 8 * s)) * (mb / 4 + s) ** (a - 1)) / (2 * (1 + a) * (2 + a))

        sigma_3max = sigma_cm * 0.47 * (sigma_cm / (ga_ma * H)) ** -0.94
        sigma_3n = sigma_3max / sigma_c

        # Erm = Ei * (0.02 + (1 - D / 2) / (1 + torch.exp((60 + 15 * D - GSI) / 11)))
        fai = torch.asin((6 * a * mb * (s + mb * sigma_3n) ** (a - 1)) / (
                2 * (1 + a) * (2 + a) + 6 * a * mb * (s + mb * sigma_3n) ** (a - 1))) * 180 / torch.pi
        c = (sigma_c * ((1 + 2 * a) * s + (1 - a) * mb * sigma_3n) * (s + mb * sigma_3n) ** (a - 1)) / (
                (1 + a) * (2 + a) * torch.sqrt(
            1 + 6 * a * mb * (s + mb * sigma_3n) ** (a - 1) / ((1 + a) * (2 + a))))
        # print(GSI,sigma_c,H,fai,c)
        return torch.cat([fai.view(-1, 1), c.view(-1, 1)], dim=1)


'''
# 第三层：调用sklearn模型
class SklearnModelLayer(nn.Module):
    def __init__(self):
        super(SklearnModelLayer, self).__init__()
        self.minmax_model=joblib.load("min_max_model_for_c&fai.pkl")
        self.models = [
            joblib.load("6model//SVM.pkl"),
            joblib.load("6model//XGBoost.pkl"),
            joblib.load("6model//RF.pkl"),
            joblib.load("6model//KNN.pkl"),
            joblib.load("6model//MLP.pkl")
        ]

    def forward(self, x):

        x_cpu = x.cpu().detach().numpy()

        y_cpu=self.minmax_model.transform(x_cpu)

        #print(y_cpu)

        column_min = np.min(y_cpu, axis=0)  # 计算每一列的最小值
        column_max = np.max(y_cpu, axis=0)  # 计算每一列的最大值
        #print(column_min,column_max)

        #print(self.models[0].predict_proba(y_cpu)[0])
        #print(self.models[1].predict_proba(y_cpu)[0])
        meta_features = np.column_stack([basemodel.predict_proba(y_cpu) for basemodel in self.models])

        #print(meta_features.shape)
        #print(meta_features[0])

        return torch.tensor(meta_features, dtype=torch.float32).to(x.device)

# 第四层：集成学习模型
class EnsembleModel(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=96, num_classes=4):#96
        super(EnsembleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        #self.softmax = nn.Softmax(dim=1)  # Softmax用于四分类任务

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        #print(x)
        #x = self.softmax(x)
        #x = torch.argmax(x, dim=1)
        return x
'''


# 组合整个模型
class FullModel(nn.Module):
    def __init__(self, input_dim=84, hidden_dim=96, output_dim=1, num_heads=12):
        super(FullModel, self).__init__()
        self.bp_network = MultiHeadAttentionANN(input_dim, hidden_dim, output_dim, num_heads,
                                                pretrained_model_path='MAT_data&res\\multihead_attention_ann.pth')
        self.custom_layer = CustomComputationLayer()
        # self.sklearn_layer = SklearnModelLayer()
        # self.ensemble_model = EnsembleModel()

    def forward(self, x):
        x_main = x[:, :84]  # 前84维
        x_aux = x[:, -2:]  # 后2维

        y1 = self.bp_network(x_main)

        x_concat = torch.cat([y1, x_aux], dim=1)  # 拼接，dim=1 表示列拼接

        y1_transformed = self.custom_layer(x_concat)

        return y1_transformed


def min_max_scale(x):
    scale = torch.tensor([0.02, 0.2])
    return x * scale
def train_model(X_train, y_train, X_test, y_test, model):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    # print(X_train_tensor.shape)
    # print(y_train_tensor.shape)
    # print(y_test_tensor)

    model.eval()
    with torch.no_grad():
        preds = model(X_train_tensor)
        yy = preds.cpu()  # .numpy()

        preds = model(X_test_tensor)
        yy_test = preds.cpu()  # .numpy()

    #minmax_model = joblib.load("STM_data/min_max_model_for_c&fai.pkl")
    pd.DataFrame(yy.numpy()).to_csv('3_GPR_data\\x_train.csv', header=None, index=False)
    pd.DataFrame(yy_test.numpy()).to_csv('3_GPR_data\\x_test.csv', header=None, index=False)
    yy2 = min_max_scale(yy)
    #print(yy2)
    yy2_test = min_max_scale(yy_test)

    files = os.listdir(os.path.join(os.getcwd(), '6model'))
    ls = ['KNN', 'LR', 'MLP', 'RF', 'SVM', 'XGBoost']
    dicts = []
    n = 0
    for i in files:
        if 'pkl' in i:
            if n == 1:
                n += 1
                continue
            file_path = "6model//%s" % i  # 替换为你的pkl文件路径
            model = load_model_from_pkl(file_path)
            dicts.append((ls[n], model))
            n += 1
    print(dicts)

    stacking_model = StackingClassifier(
        estimators=dicts,
        final_estimator=dicts[1][1])

    # 也就是说这些子模型全部是没有经过训练的

    for i in dicts:
        reeer = i[1].predict_proba(yy2)
        #print(reeer)

    stacking_model.fit(yy2, y_train)
    y_pred = stacking_model.predict(yy2)
    evaluate_model(y_train, y_pred, 'Train Set')

    y_pred = stacking_model.predict(yy2_test)
    evaluate_model(y_test, y_pred, 'Test Set')

    dff = pd.DataFrame()
    dff['Stacking'] = y_pred
    '''
    parent_dir = os.path.dirname(os.getcwd())
    dff.to_csv(os.path.join(parent_dir,'res','STM_预测结果','Stacking模型结果.csv'), index=False)

    probabity = stacking_model.predict_proba(yy2_test)
    pd.DataFrame(probabity).to_csv(os.path.join(parent_dir,'res','STM_预测结果',f'Porba_{filess}_Stacking.csv'), index=False)
    '''


if __name__ == "__main__":
    filess = 'STM_data'
    X_train = pd.read_csv(f'{filess}//X_train.csv', header=None)
    #print(X_train.shape)
    X2_train = pd.read_csv(f'{filess}/X2_train.csv', header=None)

    y_train = pd.read_csv(f'{filess}//y_train.csv', header=None)
    y2_train = pd.read_csv(f'{filess}//y2_train.csv', header=None).astype(int).to_numpy()  # 这是独热编码
    y2_train = np.argmax(y2_train, axis=1)

    X_test = pd.read_csv(f'{filess}//X_test.csv', header=None)
    X2_test = pd.read_csv(f'{filess}//X2_test.csv', header=None)

    y_test = pd.read_csv(f'{filess}//y_test.csv', header=None)
    y2_test = pd.read_csv(f'{filess}//y2_test.csv', header=None).astype(int).to_numpy()  # 这是独热编码
    y2_test = np.argmax(y2_test, axis=1)

    # 训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullModel().to(device)

    final_x_train = np.hstack([X_train.to_numpy(), X2_train.to_numpy()])
    final_y_train = y2_train.reshape(-1, 1)

    final_x_test = np.hstack([X_test.to_numpy(), X2_test.to_numpy()])
    final_y_test = y2_test.reshape(-1, 1)

    train_model(final_x_train, final_y_train, final_x_test, final_y_test, model)

