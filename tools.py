import matplotlib.pyplot as plt
import torch
import cv2
import logging
import os
import pickle

def test():
    pass


def unpickle(file):
    with open(file, 'rb') as fo:
        img_dict = pickle.load(fo)

    return img_dict


def create_logger(log_root, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fmt = '[%(asctime)s] (%(name)s): %(levelname)s %(message)s'

    file_handler = logging.FileHandler(os.path.join(log_root, f"{name}.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    return logger


def img_show(e_img, d_img):
    fig = plt.figure(figsize=(10, 7))
    rows = 1
    columns = 2

    fig.add_subplot(rows, columns, 1)
    plt.imshow(e_img)
    plt.axis('off')
    plt.title("e_img")

    fig.add_subplot(rows, columns, 2)
    plt.imshow(d_img)
    plt.axis('off')
    plt.title("d_img")

    plt.show()


def prediction(img_tensor, model, device):
    model = model.eval().to(device)
    # with open("imagenet_classes.txt", "r") as f:
    #     categories = [s.strip() for s in f.readlines()]
    with torch.no_grad():
        out = model(img_tensor)

    probabilities = torch.nn.functional.softmax(out[0], dim=0)

    # Print top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 3)

    print(top5_catid, top5_prob)

    # for i in range(top5_prob.size(0)):
    #     print(categories[top5_catid[i]], top5_prob[i].item())

    return out


def loading_img(img_path, transformer):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_tensor = transformer(img).unsqueeze(0)
    img_tensor = img_tensor
    return img_tensor

