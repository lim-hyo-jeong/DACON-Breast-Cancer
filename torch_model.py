import torch
from torch import nn
import timm


class ConvNet(nn.Module):
    def __init__(self, backbone_name, pretrained, backbone_out_dims, n_classes):
        super(ConvNet, self).__init__()

        self.backbone = timm.create_model(
            model_name=backbone_name,
            pretrained=pretrained,
            num_classes=n_classes,
        )

        self.backbone.classifier = nn.Linear(
            self.backbone.classifier.in_features, backbone_out_dims)

    def forward(self, x):
        x = self.backbone(x)
        return x


class TabularNet(nn.Module):
    def __init__(self, tabular_out_dims, n_tabulars):
        super(TabularNet, self).__init__()

        self.tabular_fc = nn.Sequential(
            nn.Linear(n_tabulars, tabular_out_dims//4),
            nn.BatchNorm1d(tabular_out_dims//4),
            nn.LeakyReLU(),
            nn.Linear(tabular_out_dims//4, tabular_out_dims//2),
            nn.BatchNorm1d(tabular_out_dims//2),
            nn.LeakyReLU(),
            nn.Linear(tabular_out_dims//2, tabular_out_dims),
            nn.BatchNorm1d(tabular_out_dims),
            nn.LeakyReLU(),
            nn.Linear(tabular_out_dims, tabular_out_dims),
        )

    def forward(self, x):
        x = self.tabular_fc(x)
        return x


class BCModel(nn.Module):
    def __init__(self, backbone_name, pretrained, backbone_out_dims, n_tabulars, tabular_out_dims, n_classes):

        super(BCModel, self).__init__()

        self.image_model = ConvNet(backbone_name=backbone_name,
                                   pretrained=pretrained,
                                   backbone_out_dims=backbone_out_dims,
                                   n_classes=n_classes)

        self.tabular_model = TabularNet(
            tabular_out_dims=tabular_out_dims, n_tabulars=n_tabulars)

        self.classifier = nn.Sequential(
            nn.Linear(backbone_out_dims + tabular_out_dims, n_classes),
            nn.Sigmoid(),
        )

    def forward(self, x, x_tab):
        x = self.image_model(x)
        x_tab = self.tabular_model(x_tab)

        x = torch.cat([x, x_tab], dim=1)
        output = self.classifier(x)
        return output


class MILTransformer(nn.Module):
    def __init__(self, backbone_name, pretrained, backbone_out_dims, n_instances, n_classes):
        super(MILTransformer, self).__init__()

        self.backbone_name = backbone_name
        self.n_instances = n_instances
        self.backbone_out_dims = backbone_out_dims

        self.backbone = timm.create_model(
            model_name=backbone_name,
            pretrained=pretrained,
            num_classes=n_classes,
        )

        if backbone_out_dims is None:
            self.in_features = self.backbone.get_classifier().in_features
            self.backbone.head = nn.Identity()
        else:
            self.backbone.head = nn.Linear(
                self.backbone.head.in_features, backbone_out_dims)

    def get_mil_out_dims(self):
        if self.backbone_out_dims is None:
            return self.in_features * self.n_instances
        else:
            return self.backbone_out_dims * self.n_instances

    def forward(self, x):
        bs, n, ch, h, w = x.shape
        x = x.view(bs * n, ch, h, w)

        x = self.backbone(x)
        emb_bs, emb_size = x.shape
        x = x.contiguous().view(bs, emb_size * n)
        return x


class BCMILModel(nn.Module):
    def __init__(self, backbone_name, pretrained, backbone_out_dims, n_instances, n_classes):

        super(BCMILModel, self).__init__()

        self.backbone_name = backbone_name

        self.mil_model = MILTransformer(backbone_name=backbone_name,
                                        pretrained=pretrained,
                                        backbone_out_dims=backbone_out_dims,
                                        n_instances=n_instances,
                                        n_classes=n_classes,
                                        )

        mil_out_dims = self.mil_model.get_mil_out_dims()

        self.classifier = nn.Sequential(
            nn.Linear(mil_out_dims, n_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.mil_model(x)
        output = self.classifier(x)
        return output
