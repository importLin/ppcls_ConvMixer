import os.path
import cv2
import os

from block_operation import ImgBlockOp
from tools import unpickle


class ImgCipher(ImgBlockOp):
    def __init__(self, block_size, keys):
        self.block_size = block_size
        self.conv_keys = keys

    def encryption(self, plaintext):
        mb_list = self.block_segmentation(plaintext, self.block_size[1])
        for m in range(len(mb_list)):
            current_key = self.conv_keys[m]
            mb = mb_list[m]
            sb_list = self.block_segmentation(mb, self.block_size[2])
            sb_list = self.block_shuffling(sb_list, current_key["sb_shuffling"])
            for s in range(len(sb_list)):
                sb = sb_list[s]
                sb = self.block_routing(sb, current_key["sb_routing"][s])
                sb = self.block_flipping(sb, current_key["sb_flipping"][s])
                sb = self.np_transformation(sb, current_key["sb_NPtrans"][s])
                sb = self.c_shuffling(sb, current_key["sb_Cshuffling"][s], 1)

                sb_list[s] = sb

            mb = self.block_intergration(sb_list, self.block_size[1])
            mb_list[m] = mb
        # main-block2img
        ciphertext = self.block_intergration(mb_list, self.block_size[0])
        return ciphertext


def img_loading(img_path):
    img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

    return rgb_img


def main():
    file_based = 1

    # block setting
    mb_size = 14
    sb_size = 7

    # key loading
    keys_nums = (224//mb_size)**2
    common_keys = [unpickle(f"key/keys_{sb_size}/0") for _ in range(keys_nums)]
    independent_keys = [unpickle(f"key/keys_{sb_size}/{i}") for i in range(keys_nums)]

    # cipher setting
    block_size = [224, mb_size, sb_size]
    com_cipher = ImgCipher(block_size, common_keys)
    ind_cipher = ImgCipher(block_size, independent_keys)

    # sample generalize
    if file_based:
        img_name = "3_miniature poodle.bmp"
        plain_img_root = f"img/sample/plain_sample/{img_name}"
        saving_root = f"img/sample/img_sb_{sb_size}"

        ori_img = img_loading(plain_img_root)
        print("ori_img saving : ", cv2.imwrite(os.path.join(saving_root, "ori_img.png"), cv2.imread(plain_img_root)))

        com_img = com_cipher.encryption(ori_img).transpose(1, 2, 0)
        save_img = cv2.cvtColor(com_img, cv2.COLOR_RGB2BGR)
        print("com_img saving : ", cv2.imwrite(os.path.join(saving_root, f"com_img_{sb_size}.png"), save_img))

        ind_img = ind_cipher.encryption(ori_img).transpose(1, 2, 0)
        save_img = cv2.cvtColor(ind_img, cv2.COLOR_RGB2BGR)
        print("ind_img saving : ", cv2.imwrite(os.path.join(saving_root, f"ind_img_{sb_size}.png"), save_img))

    # folder based generalize
    else:
        # input and output root
        plain_img_root = "img/val_data/plain_data"
        saving_root = f"img/val_data/encrypted_mb{mb_size}_sb{sb_size}"

        if not os.path.exists(saving_root):
            os.makedirs(saving_root)

        img_names = os.listdir(plain_img_root)

        # The output name is same as input one
        for i, img_name in enumerate(img_names):
            img_path = os.path.join(plain_img_root, img_name)
            print(img_path)
            img = img_loading(img_path)
            cipher_img = ind_cipher.encryption(img)

            # output saving
            save_img = cv2.cvtColor(cipher_img.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            output_path = os.path.join(saving_root, img_name)
            print(cv2.imwrite(output_path, save_img), f"saved in{output_path}")


if __name__ == '__main__':
    main()
