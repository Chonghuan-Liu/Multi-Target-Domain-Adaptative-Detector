import torch
from torch.autograd import Function
import torch.nn as nn

import torch.nn.functional as F

class GRLayer(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.alpha = 0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output = grad_outputs.neg() * ctx.alpha
        return output

def grad_reverse(x):
    return GRLayer.apply(x)


# class _ImageDA(nn.Module):
#     def __init__(self, dim):
#         super(_ImageDA, self).__init__()
#         self.dim = dim  # feat layer          256*H*W for vgg16
#         self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1, bias=True)
#         self.Conv2 = nn.Conv2d(512, 2, kernel_size=1, stride=1, bias=True)
#         self.reLu = nn.ReLU(inplace=False)
#         self.Fc1=nn.Linear(1922,500)
#         self.Fc2=nn.Linear(500,1)
#         #self.norm=nn.LayerNorm([512,31,62])
#         #self.LabelResizeLayer = ImageLabelResizeLayer()
#
#     def forward(self, x):
#         #x=self.norm(x)
#         x = grad_reverse(x)
#         x = self.reLu(self.Conv1(x))
#         x = self.Conv2(x)
#         #print(x.shape)
#         x=torch.reshape(x,(1,2,-1))
#         x=self.Fc1(x)
#         x=self.Fc2(x)
#         x=nn.functional.softmax(x,dim=1)
#         #label = self.LabelResizeLayer(x, need_backprop)
#         return x

#注意，函数的输出为一个（1，2，1）的张量，再trainer中使用交叉熵需要删除最后一个维度

#-------------------------------------------------qiu_bing_classifer
class _ImageDA(nn.Module):
    def __init__(self,input_dim):
        super(_ImageDA, self).__init__()
        #in_channels, out_channels, kernel_size, stride,\
        #padding, dilation, groups, bias
        self.da_conv_1 = nn.Conv2d(input_dim, 256, 3, 1, 1)
        self.da_conv_2 = nn.Conv2d(256, 128, 3, 1, 1)
        self.da_conv_3 = nn.Conv2d(128, 1, 1, 1)
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Sequential(
        #     nn.Linear(128, 512 // 4, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512 // 4, 512, bias=False),
        #     nn.Sigmoid()
        #     )
        self.relu = nn.ReLU()
        # self.BCELoss = nn.BCELoss()
        self.body = nn.Sequential(
            self.da_conv_1,
            self.relu,
            self.da_conv_2,
            self.relu
            )
        self.init_weight()



    def forward(self, x):
        x=grad_reverse(x)
        x = self.body(x)
        x = self.da_conv_3(x)
        x = F.sigmoid(x)
        #x = self.avg_pool(x).squeeze(2).squeeze(2)
        #x = self.fc(x)
        return x

    def init_weight(self):
        def normal_init(m, mean, stddev):
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()

        normal_init(self.da_conv_1, 0, 0.01)
        normal_init(self.da_conv_2, 0, 0.01)
        normal_init(self.da_conv_3, 0, 0.01)


#------------------------------------------------测试代码，勿删
# model=_ImageDA(1280)
# model.cuda()
# model.train()
# lr=0.01
# criterion=nn.CrossEntropyLoss()
# # criterion=criterion.cuda()
# optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9)
# for i in range(100):
#     optimizer.zero_grad()
#     input=torch.randn([1,1280,31,62],requires_grad=True)
#     input=input.cuda()
#     output=model(input)
#     output=output.squeeze(dim=2)
#     loss=criterion(output,torch.Tensor([1]).long().cuda())
#     loss.backward()
#    # optimizer.step()
#     print(output)
#----------------------------------------------------------

















