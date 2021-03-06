import torch
import torch.nn as nn
import torchvision.models as models

# this file is not used


class OurVGG(nn.Module):
    def __init__(self, model):
        super(OurVGG, self).__init__()
        self.features = model.features
        self.vgg_layer = nn.Sequential(*list(model.children()))

        self.final_relu = nn.ReLU(inplace=True)
        self.final_drop = nn.Dropout(p=0.5, inplace=True)
        self.final_fc = nn.Linear(in_features=1000, out_features=10, bias=True)

    def forward(self, x):
        x = self.vgg_layer(x)
        x = self.final_fc(self.final_drop(self.final_relu(x)))
        return x


if __name__ == '__main__':
    vgg = models.vgg19(num_classes=10)
    print(vgg)
    # model = torch.nn.parallel.DistributedDataParallel(vgg)
    # print(vgg.state_dict().keys())
    # ct = 0
    # for child in vgg.children():
    #     ct += 1
    #     if ct <= 2:
    #         for param in child.parameters():
    #             param.requires_grad = False
    #     else:
    #         for i, param in enumerate(child.parameters()):
    #             print(i, param)
    #             param.requires_grad = False
    #             print(i, param)
    # print(vgg.children())

    # for child in vgg.children():
    #     print(list(child.parameters()))