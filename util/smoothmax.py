
# encoding: utf-8
# !/usr/bin/env python3
import torch
import torch.nn as nn
import argparse


def Max(output, target):
    x = torch.abs(output - target)
    x, _ = x.max(1)
    return x.mean()


def Mean(output, target):
    x = torch.abs(output - target)
    x = x.mean(1).mean()
    return x


class MyHardSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        result = (0.2 * i + 0.5).clamp(min=0.0, max=1.0)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        result, = ctx.saved_tensors
        grad_input *= 0.2
        grad_input[result < -2.5] = 0
        grad_input[result > -2.5] = 0
        return grad_input

# https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function


class SmoothMax(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = torch.FloatTensor([alpha]).to('cuda')
        self.m = nn.Softmax(1).to('cuda')

    def forward(self, output, target):
        x = (output - target)**2
        x = torch.sqrt(x+1e-10)

        _max = (x*self.m(x)).sum(1, keepdim=True)
        _x = self.alpha * (x - _max + 1e-10)
        exp = torch.exp(self.alpha * _x)
        loss = (x*exp/x.shape[1]).sum(1)/(exp/x.shape[1]).sum(1)

        return loss.mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--alpha', type=float, default=1.0)
    args = parser.parse_args()

    output = torch.rand(32, 3000) * 30
    target = torch.rand(32, 3000) * 30

    smoothmax = MySmoothMax(1.0)

    print('==============================')
    print('Mean:', Mean(output, target))

    for alpha in [0.0, 0.1, 1.0, 5, 10]:
        smoothmax = SmoothMax(alpha)
        smoothmax.alpha = smoothmax.alpha.to('cpu')
        smoothmax.m = smoothmax.m.to('cpu')
        print('Alpha=', str(alpha), smoothmax(output, target))

    print('Max:', Max(output, target))

    for mul in [1e-10, 1e-5, 1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3, 1e5, 1e10]:
        print('==============================')
        print(f'Mul={mul}')
        print('Mean:', Mean(output*mul, target*mul))
        for alpha in [0.0, 1e-3, 5*1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100]:
            smoothmax = SmoothMax(alpha)
            smoothmax.alpha = smoothmax.alpha.to('cpu')
            smoothmax.m = smoothmax.m.to('cpu')
            print('Alpha=', str(alpha), smoothmax(output*mul, target*mul))

        print('Max:', Max(output*mul, target*mul))
