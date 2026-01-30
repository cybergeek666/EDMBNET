import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from lib.model_arch import modality_drop, unbalance_modality_drop


class GraphConvolution(nn.Module):
    """简化的图卷积层，参考MOGONET"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        # 使用Xavier初始化
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        # Use regular matrix multiplication instead of sparse to avoid gradient issues
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class SimpleGCN(nn.Module):
    """简化的GCN编码器，参考MOGONET的GCN_E"""
    def __init__(self, in_dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.leaky_relu(x, 0.25)
        return x


class SimpleVCDN(nn.Module):
    """简化的视图相关性发现网络，参考MOGONET的VCDN"""
    def __init__(self, num_view, num_cls, hidden_dim=128):
        super().__init__()
        self.num_cls = num_cls
        self.num_view = num_view

        # 简化的融合策略：使用注意力机制
        self.attention_weights = nn.Parameter(torch.ones(num_view, 1))
        self.fusion_layer = nn.Sequential(
            nn.Linear(num_cls, hidden_dim),
            nn.LeakyReLU(0.25),
            nn.Linear(hidden_dim, num_cls)
        )

        # 初始化
        nn.init.xavier_normal_(self.attention_weights.data)

    def forward(self, in_list):
        num_view = len(in_list)

        # 使用注意力加权融合
        attention_weights = self.attention_weights.squeeze()
        if attention_weights.dim() == 0:  # 如果是标量
            attention_weights = attention_weights.unsqueeze(0)
        weights = F.softmax(attention_weights, dim=0)
        fused_features = None

        for i in range(num_view):
            weighted_feat = weights[i] * torch.sigmoid(in_list[i])
            if fused_features is None:
                fused_features = weighted_feat
            else:
                fused_features = fused_features + weighted_feat

        output = self.fusion_layer(fused_features)
        # 确保返回的是张量
        if not isinstance(output, torch.Tensor):
            output = torch.tensor(output)
        return output


class BRCA_Baseline_Simple(nn.Module):
    """简化的BRCA基线模型，借鉴MOGONET的设计思想"""
    def __init__(self, args, input_dims=[1000, 1000, 503], hidden_dim=256, num_classes=5):
        super().__init__()
        self.args = args
        self.num_classes = num_classes

        # 创建邻接矩阵参数（参考MOGONET的设置）
        self.adj_parameter = 10  # BRCA数据集的推荐值

        # 模态编码器 - 简化为单层
        self.modal_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dims[i], hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ) for i in range(3)
        ])

        # 简化的GCN处理
        self.gcn_encoders = nn.ModuleList([
            SimpleGCN(hidden_dim, hidden_dim, dropout=0.5) for _ in range(3)
        ])

        # 分类器
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(3)
        ])

        # 简化的多模态融合器
        self.fusion_net = SimpleVCDN(3, num_classes, hidden_dim=128)

    def build_adj_matrix(self, features, adj_parameter=10):
        """构建邻接矩阵，参考MOGONET的方法"""
        device = features.device

        # 计算余弦距离
        features_norm = F.normalize(features, p=2, dim=1)
        similarity = torch.mm(features_norm, features_norm.t())

        # 构建邻接矩阵
        adj = torch.zeros_like(similarity)
        for i in range(features.size(0)):
            # 找到最相似的k个邻居
            k = min(adj_parameter, features.size(0) - 1)
            _, indices = torch.topk(similarity[i], k + 1)
            adj[i, indices[1:]] = similarity[i, indices[1:]]  # 排除自己

        # 添加自环并归一化
        adj = adj + torch.eye(adj.size(0), device=device)
        adj = F.normalize(adj, p=1, dim=1)

        return adj

    def forward(self, x1, x2, x3):
        batch_size = x1.size(0)

        # 1. 模态编码
        modal_features = []
        for i, (encoder, x) in enumerate(zip(self.modal_encoders, [x1, x2, x3])):
            feat = encoder(x)
            modal_features.append(feat)
            # print(f"Modal {i+1} after encoding: {feat.shape}")  # Debug info removed

        # 2. 构建邻接矩阵并应用GCN
        gcn_features = []
        for i, (gcn, feat) in enumerate(zip(self.gcn_encoders, modal_features)):
            adj = self.build_adj_matrix(feat, self.adj_parameter)
            gcn_feat = gcn(feat, adj)
            gcn_features.append(gcn_feat)
            # print(f"Modal {i+1} after GCN: {gcn_feat.shape}")  # Debug info removed

        # 3. 独立分类器
        classifier_outputs = []
        for i, (classifier, feat) in enumerate(zip(self.classifiers, gcn_features)):
            output = classifier(feat)
            classifier_outputs.append(output)
            # print(f"Modal {i+1} classifier output: {output.shape}")  # Debug info removed

        # 4. 多模态融合
        if len(classifier_outputs) >= 2:
            fused_output = self.fusion_net(classifier_outputs)
            # print(f"Fused output: {fused_output.shape}")  # Debug info removed
        else:
            fused_output = classifier_outputs[0]

        # 确保返回的是张量
        if not isinstance(fused_output, torch.Tensor):
            fused_output = torch.tensor(fused_output)

        return fused_output


class BRCA_Multi(nn.Module):
    """BRCA 多模态癌症亚型分类模型"""

    def __init__(self, args, input_dims=[1000, 1000, 503], hidden_dim=512, num_classes=5):
        """
        Args:
            args: 配置参数
            input_dims: 每个模态的输入特征维度列表
            hidden_dim: 隐藏层维度
            num_classes: 分类数量
        """
        super().__init__()

        self.args = args
        self.num_classes = num_classes
        self.input_dims = input_dims

        # 三个模态的编码器，使用不同的输入维度
        self.modal_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dims[i], hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for i in range(3)
        ])

        # 共享的融合层 - 动态调整输入维度
        fusion_input_dim = hidden_dim // 2 * 3
        self.shared_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 模态dropout参数
        self.p = args.p if hasattr(args, 'p') else [0, 0, 0]
        self.drop_mode = args.drop_mode if hasattr(args, 'drop_mode') else 'average'

    def forward(self, modal_1, modal_2, modal_3):
        """
        Args:
            modal_1, modal_2, modal_3: 三个模态的特征向量
        """

        # 分别编码每个模态
        x1 = self.modal_encoders[0](modal_1)
        x2 = self.modal_encoders[1](modal_2)
        x3 = self.modal_encoders[2](modal_3)

        # 应用模态dropout
        if self.drop_mode == 'average':
            x1, x2, x3, p = modality_drop(x1, x2, x3, self.p, self.args)
        else:
            x1, x2, x3, p = unbalance_modality_drop(x1, x2, x3, self.p, self.args)

        # 特征融合 - 正确处理维度
        if len(x1.shape) == 4:  # 如果是4维特征，flatten
            x1 = x1.view(x1.size(0), -1)
        if len(x2.shape) == 4:
            x2 = x2.view(x2.size(0), -1)
        if len(x3.shape) == 4:
            x3 = x3.view(x3.size(0), -1)

        # 改进的多模态融合策略
        # 1. 自适应权重融合
        x_concat = torch.cat([x1, x2, x3], dim=1)
        print(f"Concatenated feature shape: {x_concat.shape}")  # Debug

        # 2. 改进的降维策略 - 使用分层降维
        if x_concat.shape[1] > 768:
            # 第一阶段：按模态分组并降维
            x_reshaped = x_concat.view(x_concat.size(0), 3, -1)  # [batch, 3, features_per_modal]

            # 对每个模态单独进行降维
            modal_features = []
            for i in range(3):
                modal_feat = x_reshaped[:, i, :]  # [batch, features_per_modal]

                # 使用更智能的降维策略
                if modal_feat.shape[1] > 768:
                    # 第一步：全局平均池化
                    modal_pooled = torch.mean(modal_feat.view(modal_feat.size(0), -1, 256), dim=1)

                    # 第二步：如果仍然过大，使用自适应截断
                    if modal_pooled.shape[1] > 256:
                        # 计算特征重要性（使用L2范数）
                        feat_norm = torch.norm(modal_pooled, dim=1, keepdim=True)
                        importance = feat_norm / torch.sum(feat_norm, dim=1, keepdim=True)

                        # 保留最重要的256维特征
                        _, top_indices = torch.topk(importance.squeeze(), 256, dim=1)
                        modal_reduced = torch.gather(modal_pooled, 1,
                                                   top_indices.unsqueeze(-1).expand(-1, -1, modal_pooled.size(-1)))
                        modal_features.append(modal_reduced)
                    else:
                        modal_features.append(modal_pooled)
                else:
                    modal_features.append(modal_feat)

            # 拼接所有模态的特征
            x_fused = torch.cat(modal_features, dim=1)
            print(f"After improved fusion: {x_fused.shape}")  # Debug

            # 最终调整到768维
            if x_fused.shape[1] != 768:
                if x_fused.shape[1] > 768:
                    x_fused = x_fused[:, :768]  # 截断
                else:
                    # 填充到768维
                    padding = 768 - x_fused.shape[1]
                    x_fused = torch.cat([x_fused, torch.zeros(x_fused.size(0), padding, device=x_fused.device)], dim=1)
        else:
            x_fused = x_concat

        # 3. 最终分类
        output = self.shared_fusion(x_fused)

        return output


class BRCA_Baseline(nn.Module):
    """BRCA 基线模型"""

    def __init__(self, args, input_dim=2000, hidden_dim=512, num_classes=5):
        super().__init__()

        self.args = args
        self.num_classes = num_classes

        # 支持单个input_dim或input_dims列表
        if isinstance(input_dim, list):
            input_dims = input_dim
        else:
            input_dims = [input_dim, input_dim, input_dim]  # 假设所有模态维度相同

        # 三个模态的编码器 - 优化架构，减少参数量
        hidden_dim = 256  # 减少隐藏层维度
        self.modal_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dims[i], hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),  # 添加BatchNorm
                nn.Dropout(0.3),  # 减少dropout
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.Dropout(0.2)
            ) for i in range(3)
        ])

        # 共享的融合层 - 动态调整输入维度
        self.shared_fusion = nn.Sequential(
            nn.Linear(768, hidden_dim),  # 固定输入为768维
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 模态dropout参数
        self.p = args.p if hasattr(args, 'p') else [0, 0, 0]
        self.drop_mode = args.drop_mode if hasattr(args, 'drop_mode') else 'average'

    def forward(self, modal_1, modal_2, modal_3):
        # 分别编码每个模态
        x1 = self.modal_encoders[0](modal_1)
        x2 = self.modal_encoders[1](modal_2)
        x3 = self.modal_encoders[2](modal_3)

        # 应用模态dropout
        if self.drop_mode == 'average':
            x1, x2, x3, p = modality_drop(x1, x2, x3, self.p, self.args)
        else:
            x1, x2, x3, p = unbalance_modality_drop(x1, x2, x3, self.p, self.args)

        # 特征融合 - 正确处理维度
        if len(x1.shape) == 4:  # 如果是4维特征，flatten
            x1 = x1.view(x1.size(0), -1)
        if len(x2.shape) == 4:
            x2 = x2.view(x2.size(0), -1)
        if len(x3.shape) == 4:
            x3 = x3.view(x3.size(0), -1)

        # 改进的多模态融合策略
        # 1. 自适应权重融合
        x_concat = torch.cat([x1, x2, x3], dim=1)
        print(f"Concatenated feature shape: {x_concat.shape}")  # Debug

        # 2. 改进的降维策略 - 使用分层降维
        if x_concat.shape[1] > 768:
            # 第一阶段：按模态分组并降维
            x_reshaped = x_concat.view(x_concat.size(0), 3, -1)  # [batch, 3, features_per_modal]

            # 对每个模态单独进行降维
            modal_features = []
            for i in range(3):
                modal_feat = x_reshaped[:, i, :]  # [batch, features_per_modal]

                # 使用更智能的降维策略
                if modal_feat.shape[1] > 768:
                    # 第一步：全局平均池化
                    modal_pooled = torch.mean(modal_feat.view(modal_feat.size(0), -1, 256), dim=1)

                    # 第二步：如果仍然过大，使用自适应截断
                    if modal_pooled.shape[1] > 256:
                        # 计算特征重要性（使用L2范数）
                        feat_norm = torch.norm(modal_pooled, dim=1, keepdim=True)
                        importance = feat_norm / torch.sum(feat_norm, dim=1, keepdim=True)

                        # 保留最重要的256维特征
                        _, top_indices = torch.topk(importance.squeeze(), 256, dim=1)
                        modal_reduced = torch.gather(modal_pooled, 1,
                                                   top_indices.unsqueeze(-1).expand(-1, -1, modal_pooled.size(-1)))
                        modal_features.append(modal_reduced)
                    else:
                        modal_features.append(modal_pooled)
                else:
                    modal_features.append(modal_feat)

            # 拼接所有模态的特征
            x_fused = torch.cat(modal_features, dim=1)
            print(f"After improved fusion: {x_fused.shape}")  # Debug

            # 最终调整到768维
            if x_fused.shape[1] != 768:
                if x_fused.shape[1] > 768:
                    x_fused = x_fused[:, :768]  # 截断
                else:
                    # 填充到768维
                    padding = 768 - x_fused.shape[1]
                    x_fused = torch.cat([x_fused, torch.zeros(x_fused.size(0), padding, device=x_fused.device)], dim=1)
        else:
            x_fused = x_concat

        # 3. 最终分类
        output = self.shared_fusion(x_fused)

        return output


class BRCA_Baseline_Auxi(nn.Module):
    """BRCA 带有辅助分支的模型"""

    def __init__(self, args, input_dim=2000, hidden_dim=512, num_classes=5):
        super().__init__()

        self.args = args
        self.num_classes = num_classes

        # 三个模态的编码器
        self.modal_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for _ in range(3)
        ])

        # 辅助分支编码器
        self.aux_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 共享的融合层 - 动态调整输入维度
        fusion_input_dim = hidden_dim // 2 * 3
        self.shared_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 模态dropout参数
        self.p = args.p if hasattr(args, 'p') else [0, 0, 0]
        self.drop_mode = args.drop_mode if hasattr(args, 'drop_mode') else 'average'

    def forward(self, modal_1, modal_2, modal_3):
        # 分别编码每个模态
        x1 = self.modal_encoders[0](modal_1)
        x2 = self.modal_encoders[1](modal_2)
        x3 = self.modal_encoders[2](modal_3)

        # 辅助分支输出
        aux_out1 = self.aux_encoder(x1)
        aux_out2 = self.aux_encoder(x2)
        aux_out3 = self.aux_encoder(x3)

        # 应用模态dropout
        if self.drop_mode == 'average':
            x1, x2, x3, p = modality_drop(x1, x2, x3, self.p, self.args)
        else:
            x1, x2, x3, p = unbalance_modality_drop(x1, x2, x3, self.p, self.args)

        # 特征融合 - 正确处理维度
        if len(x1.shape) == 4:  # 如果是4维特征，flatten
            x1 = x1.view(x1.size(0), -1)
        if len(x2.shape) == 4:
            x2 = x2.view(x2.size(0), -1)
        if len(x3.shape) == 4:
            x3 = x3.view(x3.size(0), -1)

        # 改进的多模态融合策略
        # 1. 自适应权重融合
        x_concat = torch.cat([x1, x2, x3], dim=1)
        print(f"Concatenated feature shape: {x_concat.shape}")  # Debug

        # 2. 改进的降维策略 - 使用分层降维
        if x_concat.shape[1] > 768:
            # 第一阶段：按模态分组并降维
            x_reshaped = x_concat.view(x_concat.size(0), 3, -1)  # [batch, 3, features_per_modal]

            # 对每个模态单独进行降维
            modal_features = []
            for i in range(3):
                modal_feat = x_reshaped[:, i, :]  # [batch, features_per_modal]

                # 使用更智能的降维策略
                if modal_feat.shape[1] > 768:
                    # 第一步：全局平均池化
                    modal_pooled = torch.mean(modal_feat.view(modal_feat.size(0), -1, 256), dim=1)

                    # 第二步：如果仍然过大，使用自适应截断
                    if modal_pooled.shape[1] > 256:
                        # 计算特征重要性（使用L2范数）
                        feat_norm = torch.norm(modal_pooled, dim=1, keepdim=True)
                        importance = feat_norm / torch.sum(feat_norm, dim=1, keepdim=True)

                        # 保留最重要的256维特征
                        _, top_indices = torch.topk(importance.squeeze(), 256, dim=1)
                        modal_reduced = torch.gather(modal_pooled, 1,
                                                   top_indices.unsqueeze(-1).expand(-1, -1, modal_pooled.size(-1)))
                        modal_features.append(modal_reduced)
                    else:
                        modal_features.append(modal_pooled)
                else:
                    modal_features.append(modal_feat)

            # 拼接所有模态的特征
            x_fused = torch.cat(modal_features, dim=1)
            print(f"After improved fusion: {x_fused.shape}")  # Debug

            # 最终调整到768维
            if x_fused.shape[1] != 768:
                if x_fused.shape[1] > 768:
                    x_fused = x_fused[:, :768]  # 截断
                else:
                    # 填充到768维
                    padding = 768 - x_fused.shape[1]
                    x_fused = torch.cat([x_fused, torch.zeros(x_fused.size(0), padding, device=x_fused.device)], dim=1)
        else:
            x_fused = x_concat

        # 3. 最终分类
        output = self.shared_fusion(x_fused)

        return output, [aux_out1, aux_out2, aux_out3]
