import torch
import torch.nn as nn
from torchvision.models import alexnet

from relative_similarity import RelativeSimilarity

class AlexNet(nn.Module):
    def __init__(self,
                 nbit, nclass, batchsize,
                 init_method='M',
                 pretrained=True, freeze_weight=False,
                 device='cuda',
                 **kwargs):
        super(AlexNet, self).__init__()

        model = alexnet(pretrained=pretrained)
        self.features = model.features
        self.avgpool = model.avgpool
        fc = []
        for i in range(6):
            fc.append(model.classifier[i])
        self.fc = nn.Sequential(*fc)

        if freeze_weight:
            for param in self.features.parameters():
                param.requires_grad_(False)
            for param in self.fc.parameters():
                param.requires_grad_(False)

        self.hash_fc = nn.Sequential(
            nn.Linear(model.classifier[6].in_features, nbit, bias=False),
            nn.BatchNorm1d(nbit, momentum=0.1)
        )
        nn.init.normal_(self.hash_fc[0].weight, std=0.01)

        self.relative_similarity = RelativeSimilarity(nbit, nclass, batchsize, init_method=init_method, device=device)

    def get_backbone_params(self):
        return list(self.features.parameters()) + list(self.fc.parameters())

    def get_hash_params(self):
        return list(self.relative_similarity.parameters()) + list(self.hash_fc.parameters())

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        z = self.hash_fc(x)
        logits = self.relative_similarity(z)
        return logits, z