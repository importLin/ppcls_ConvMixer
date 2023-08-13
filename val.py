import os
import timm
import torch

from timm.utils import accuracy, AverageMeter
from torch.utils.data import DataLoader, Dataset
from data_loader import get_transformer, Cifar10
from tools import unpickle, create_logger
from DynamicConv import conv_initialize, pos_initialize


def val_one_epoch(model, data_loader, logger, device):
    print("testing mode activated!")
    model.eval().to(device)
    acc_meter = AverageMeter()
    with torch.no_grad():
        for i, (img_tensor, labels) in enumerate(data_loader):
            img_tensor = img_tensor.to(device)
            labels = labels.to(device)

            out = model(img_tensor)
            batch_acc = accuracy(out, labels, topk=(1,))
            logger.info(f"Batch acc-{acc_meter.avg:.2f}")
            acc_meter.update(batch_acc[0].item(), labels.size(0))

    logger.info(f"Final acc-{acc_meter.avg:.2f}")
    return acc_meter.avg


def dict_remapping(ori_dict):
    new_dict = {}
    weights_dict = ori_dict["weights"]
    for k in weights_dict.keys():
        new_k = k.split("module.")[1]
        new_dict[new_k] = weights_dict[k]
    return new_dict


def main():
    encrypted_model_used = 1
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img_transformer = get_transformer()
    img_size = 224
    patch_size = 16
    sb_size = 16
    out_channels = 768

    # root
    pretrained_model_path = "weights/baseline/baseline_acc9712.pth"
    log_root = "log"

    # encrypted model used
    if encrypted_model_used:
        encrypted_img_root = f"img/val_data/encrypted_mb{patch_size}_sb{sb_size}"
        encrypted_weights_root = f"weights/ind_sb_{sb_size}"

        # model initialize
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)

        # remapping the pretrained dict to format of timm
        pretrained_dict = torch.load(pretrained_model_path)
        # timm_dict = dict_remapping(pretrained_dict)
        model.load_state_dict(pretrained_dict)

        # loading encrypted weights
        patch_pos = unpickle(f"key/keys_{sb_size}/0")["mb_shuffling"]
        model.patch_embed.proj = conv_initialize(patch_pos, out_channels, img_size, patch_size, encrypted_weights_root)
        model.pos_embed = pos_initialize(encrypted_weights_root)

        testing_set = Cifar10(encrypted_img_root, img_transformer)
        logger = create_logger(log_root, "encrypted_ViT")

    # baseline used
    else:
        plain_img_root = "img/val_data/plain_data"
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
        # remapping the pretrained dict to timm format
        pretrained_dict = torch.load(pretrained_model_path)
        timm_dict = dict_remapping(pretrained_dict)
        model.load_state_dict(timm_dict)

        testing_set = Cifar10(plain_img_root, img_transformer)
        logger = create_logger(log_root, "original_ViT")

    testing_loader = DataLoader(
        testing_set,
        batch_size=500,
        shuffle=True,
        drop_last=False
    )
    for epoch in range(1):
        acc = val_one_epoch(model, testing_loader, logger, device)


if __name__ == '__main__':
    main()
