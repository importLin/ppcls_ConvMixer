import torch
import timm
import os
import numpy as np

from tools import unpickle
from block_operation import ModelBlockOp
from copy import deepcopy
from collections import OrderedDict


def dict_remapping(ori_dict):
    new_dict = {}
    weights_dict = ori_dict["weights"]
    for k in weights_dict.keys():
        new_k = k.split("module.")[1]
        new_dict[new_k] = weights_dict[k]
    return new_dict


class ModelCipher(ModelBlockOp):
    def __init__(self, block_size):
        # [3,224,224]
        self.block_size = block_size

    def pe_encryption(self, pe_weight, current_key):
        sb_list = self.block_segmentation(pe_weight, self.block_size[1])
        sb_list = self.block_shuffling(sb_list, current_key["sb_shuffling"])
        for s in range(len(sb_list)):
            sb = sb_list[s]

            sb = self.block_routing(sb, current_key["sb_routing"][s])
            sb = self.block_flipping(sb, current_key["sb_flipping"][s])
            sb = self.np_transformation(sb, current_key["sb_NPtrans"][s])
            sb = self.c_shuffling(sb, current_key["sb_Cshuffling"][s], 1)

            sb_list[s] = sb

        # pe_weight.shape
        cipher_pe = self.block_intergration(sb_list, pe_weight.shape)
        return cipher_pe


def main():
    input_size = 224
    mb_size = 14
    sb_size = 7

    block_size = [mb_size, sb_size]
    patch_nums = (input_size // mb_size) ** 2
    model_cipher = ModelCipher(block_size)

    # root
    key_root = f"key/keys_{sb_size}"
    pretrained_model_path = None
    saving_root = f"weights/ind_sb_{sb_size}"

    # keys_for_fig loading
    keys_list = [unpickle(os.path.join(key_root, f"{i}")) for i in range(patch_nums)]

    # loading fine-tuning model weights
    if pretrained_model_path:
        print("using pretrain weights for encryption")
        model = timm.create_model("convmixer_1024_20_ks9_p14", pretrained=True, num_classes=10)
        pretrained_dict = torch.load(pretrained_model_path)
        timm_dict = dict_remapping(pretrained_dict)
        model.load_state_dict(timm_dict)

        pe = model.stem
    # loading timm model weights
    else:
        model = timm.create_model("convmixer_1024_20_ks9_p14", pretrained=False)
        pe = model.stem

    for i in range(patch_nums):
        weights_dict = deepcopy(pe).state_dict()
        weights_dict["0.weight"].data = model_cipher.pe_encryption(weights_dict["0.weight"], keys_list[i])

        torch.save(weights_dict, os.path.join(saving_root, f"{i}.pth"))
        print("Encrypted PE weights have saved in:", os.path.join(saving_root, f"{i}.pth"))


if __name__ == '__main__':
    main()
