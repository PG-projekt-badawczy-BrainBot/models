import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEnhancement(nn.Module):
    """
    Enhances features by adding sum and difference of symmetric channel pairs.
    Args:
        symmetric_pairs (list of tuple): List of tuples indicating indices of symmetric channels, e.g., [(i, j), ...]
    """
    def __init__(self, symmetric_pairs):
        super(FeatureEnhancement, self).__init__()
        self.pairs = symmetric_pairs

    def forward(self, x):
        # x: (batch, C, T)
        enh = [x]
        for i, j in self.pairs:
            diff = x[:, i, :] - x[:, j, :]
            summ = x[:, i, :] + x[:, j, :]
            enh.append(diff.unsqueeze(1))
            enh.append(summ.unsqueeze(1))
        x_enh = torch.cat(enh, dim=1)
        return x_enh


class M1DCNN(nn.Module):
    """
    Multi-scale 1D temporal convolution block with explicit padding to preserve time dimension.
    """
    def __init__(self, in_channels, n_filters=10, dropout=0.25):
        super(M1DCNN, self).__init__()
        ks = [40, 70, 85]
        self.branches = nn.ModuleList()
        for k in ks:
            pad_l = (k - 1) // 2
            pad_r = (k - 1) - pad_l
            branch = nn.Sequential(
                nn.ConstantPad1d((pad_l, pad_r), 0),
                nn.Conv1d(in_channels, n_filters, kernel_size=k, padding=0),
                nn.BatchNorm1d(n_filters),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1),
                nn.BatchNorm1d(n_filters),
                nn.Dropout(dropout),
                nn.ReLU()
            )
            self.branches.append(branch)

    def forward(self, x):
        # x: (batch, C, T)
        outs = [branch(x) for branch in self.branches]
        # Concatenate along time dimension
        return torch.cat(outs, dim=2)


class M2DCNN(nn.Module):
    """
    Multi-scale 2D spatio-temporal convolution block.
    """
    def __init__(self, time_dim, space_dim, n_filters=10, dropout=0.25):
        super(M2DCNN, self).__init__()
        ks = [45, 60, 90]
        self.branches = nn.ModuleList()
        for k in ks:
            pad_top = (k - 1) // 2
            pad_bottom = (k - 1) - pad_top
            self.branches.append(nn.Sequential(
                nn.ZeroPad2d((0, 0, pad_top, pad_bottom)),
                nn.Conv2d(1, n_filters, kernel_size=(k, 1), padding=0),
                nn.BatchNorm2d(n_filters),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Conv2d(n_filters, n_filters, kernel_size=(1, space_dim), padding=0),
                nn.BatchNorm2d(n_filters),
                nn.Dropout(dropout),
                nn.ReLU()
            ))

    def forward(self, x):
        # x: (batch, 1, T, C)
        outs = []
        for branch in self.branches:
            o = branch(x)
            # o: (batch, n_filters, T, 1) -> squeeze last dim
            o = o.squeeze(-1)
            outs.append(o)
        # Concatenate along time dimension
        return torch.cat(outs, dim=2)


class MSHCNN(nn.Module):
    def __init__(self, n_channels, time_points, symmetric_pairs, n_classes=2,
                 n_filters=10, dropout=0.25):
        super(MSHCNN, self).__init__()
        # Feature enhancement
        self.feature_enh = FeatureEnhancement(symmetric_pairs)
        enh_channels = n_channels + 2 * len(symmetric_pairs)

        # 1D and 2D branches
        self.m1d = M1DCNN(enh_channels, n_filters, dropout)
        self.m2d = M2DCNN(time_points, enh_channels, n_filters, dropout)

        # Fusion normalization
        self.bn_fusion = nn.BatchNorm1d(n_filters)
        # Pooling and classifier
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(n_filters, 100)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(100, n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (batch, C, T)
        x_enh = self.feature_enh(x)

        # 1D branch
        r1 = self.m1d(x_enh)

        # 2D branch
        b, c, t = x_enh.size()
        x2 = x_enh.view(b, 1, t, c)
        r2 = self.m2d(x2)

        # Fusion
        r = torch.cat([r1, r2], dim=2)
        r = self.bn_fusion(r)

        p = self.avg_pool(r).squeeze(-1)
        out = F.relu(self.fc1(p))
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits
