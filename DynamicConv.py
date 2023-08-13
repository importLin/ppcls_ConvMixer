import torch.nn.functional as F
import torch
import timm
import os

from torch import nn
from torchvision import transforms as T
from tools import prediction, loading_img
from tools import unpickle


def path_testing():
    print("this function is build from here! ")


def block_shuffling(block_list, key_seq):
    shuffling_list = []

    for i in range(len(block_list)):
        block_index = key_seq[i]
        shuffling_list.append(block_list[block_index])

    return shuffling_list


class DynaMicConv(nn.Module):
    def __init__(self, out_channels, weight_list, bias_list, img_size, patch_size):
        super(DynaMicConv, self).__init__()
        self.patch_nums = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.img_size = img_size
        self.out_chennels = out_channels

        self.weight_list = nn.ParameterList(
            [nn.Parameter(w) for _, w in enumerate(weight_list)]
        )
        self.bias_list = nn.ParameterList(
            [nn.Parameter(b) for _, b in enumerate(bias_list)]
        )

    def forward(self, x):
        # b,c,h,w
        out_shape = (-1, self.out_chennels, self.img_size // self.patch_size, self.img_size // self.patch_size)
        out_list = []
        p_idx = 0
        for i in range(self.img_size // self.patch_size):
            for j in range(self.img_size // self.patch_size):
                conv_w = self.weight_list[p_idx]
                conv_b = self.bias_list[p_idx]

                patch = x[:, :, self.patch_size * i: self.patch_size + (self.patch_size * i),
                        self.patch_size * j: self.patch_size + (self.patch_size * j)]
                out = F.conv2d(patch, conv_w, conv_b,
                               stride=self.patch_size, padding=0)

                out_list.append(out)
                p_idx += 1

        final_out = torch.cat(out_list, dim=2).reshape(out_shape)
        return final_out


def conv_initialize(out_channels, img_size, patch_size, weights_root):
    weight_list = []
    bias_list = []
    kernel_nums = (img_size // patch_size)**2
    for i in range(kernel_nums):
        conv_dict = torch.load(os.path.join(weights_root, f"{i}.pth"))
        weight_list.append(conv_dict["0.weight"].data)
        bias_list.append(conv_dict["0.bias"].data)
    # remapping weight_list
    return DynaMicConv(out_channels, weight_list, bias_list, img_size, patch_size)


def main():
    device = "cpu"
    img_size = 224
    patch_size = 14
    out_channels = 1024
    weights_root = "weights/ind_sb_7"

    ori_model = timm.create_model("convmixer_1024_20_ks9_p14", pretrained=True)
    ind_model = timm.create_model("convmixer_1024_20_ks9_p14", pretrained=True)
    ind_model.stem = conv_initialize(out_channels, img_size, patch_size, weights_root)

    transformer = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=torch.tensor([0.5000, 0.5000, 0.5000]),
                    std=torch.tensor([0.5000, 0.5000, 0.5000]))
    ])
    ori_x = loading_img("img/sample/plain_sample/3_miniature poodle.png", transformer).to(device)
    ind_x = loading_img("img/sample/img_sb_7/ind_img_7.png", transformer).to(device)

    print("original model output:")
    ori_out = prediction(ori_x, ori_model, device)
    print("\nencrypted model output:")
    ind_out = prediction(ind_x, ind_model, device)
    print(torch.sum(ori_out - ind_out))


if __name__ == '__main__':
    main()
