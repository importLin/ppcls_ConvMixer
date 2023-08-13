import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_line_histogram(image, title):
    r_channel = image[:, :, 0].ravel()
    g_channel = image[:, :, 1].ravel()
    b_channel = image[:, :, 2].ravel()

    plt.hist(r_channel, bins=256, range=(0, 256), color='red', alpha=0.5, label='Red')
    plt.hist(g_channel, bins=256, range=(0, 256), color='green', alpha=0.5, label='Green')
    plt.hist(b_channel, bins=256, range=(0, 256), color='blue', alpha=0.5, label='Blue')

    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()

    plt.savefig(f"fig_results/{title}.png", dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()  # Close the plot to release resources (optional)


def plot_stacked_histogram(image, title):
    r_channel = image[:, :, 0].ravel()
    g_channel = image[:, :, 1].ravel()
    b_channel = image[:, :, 2].ravel()

    plt.hist([r_channel, g_channel, b_channel], bins=256, range=(0, 256), color=['red', 'green', 'blue'], alpha=0.7,
             label=['Red', 'Green', 'Blue'])

    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title(f'Stacked Histogram({title})')
    plt.legend()
    plt.show()


def plot_side_by_side_histogram(image, title):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    r_channel = image[:, :, 0].ravel()
    g_channel = image[:, :, 1].ravel()
    b_channel = image[:, :, 2].ravel()

    ax1.hist(r_channel, bins=256, range=(0, 256), color='red', alpha=0.7, label='Red')
    ax1.set_title('Red Channel')

    ax2.hist(g_channel, bins=256, range=(0, 256), color='green', alpha=0.7, label='Green')
    ax2.set_title('Green Channel')

    ax3.hist(b_channel, bins=256, range=(0, 256), color='blue', alpha=0.7, label='Blue')
    ax3.set_title('Blue Channel')

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.legend()

    plt.suptitle(f'Side-by-Side Histograms({title})')
    plt.show()


def plot_3d_histogram(image, title):
    r_channel = image[:, :, 0].ravel()
    g_channel = image[:, :, 1].ravel()
    b_channel = image[:, :, 2].ravel()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    hist, xedges, yedges = np.histogram2d(r_channel, g_channel, bins=256, range=[[0, 256], [0, 256]])
    xidx, yidx = np.meshgrid(xedges[:-1], yedges[:-1])
    xidx = xidx.ravel()
    yidx = yidx.ravel()
    zidx = 0

    dx = dy = 1
    dz = hist.ravel()

    ax.bar3d(xidx, yidx, zidx, dx, dy, dz)

    ax.set_xlabel('Red Channel')
    ax.set_ylabel('Green Channel')
    ax.set_zlabel('Frequency')
    ax.set_title(f'3D Histogram({title})')

    plt.show()


def plot_colored_histogram(image, title):
    r_channel = image[:, :, 0].ravel()
    g_channel = image[:, :, 1].ravel()
    b_channel = image[:, :, 2].ravel()

    plt.hist(r_channel, bins=256, range=(0, 256), color='red', alpha=0.5, label='Red')
    plt.hist(g_channel, bins=256, range=(0, 256), color='green', alpha=0.5, label='Green')
    plt.hist(b_channel, bins=256, range=(0, 256), color='blue', alpha=0.5, label='Blue')

    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title(f'Colored Histogram({title})')
    plt.legend()
    plt.show()


def plot_boxplot(image, title):
    r_channel = image[:, :, 0].ravel()
    g_channel = image[:, :, 1].ravel()
    b_channel = image[:, :, 2].ravel()

    data = [r_channel, g_channel, b_channel]
    labels = ['Red', 'Green', 'Blue']

    plt.boxplot(data, labels=labels)
    plt.xlabel('Channel')
    plt.ylabel('Pixel Value')
    plt.title(f'Boxplot({title})')
    plt.show()


def cv2_to_rgb(image_cv2):
    # Convert BGR image from cv2 to RGB format
    return


def plot_histograms(image, title):
    # Convert BGR image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Call the different histogram plotting functions
    plot_line_histogram(image_rgb, title)
    # plot_stacked_histogram(image_rgb, title)
    # plot_side_by_side_histogram(image_rgb, title)
    # plot_3d_histogram(image_rgb)
    # plot_boxplot(image_rgb, title)


def calculate_rgb_image_entropy(image):
    # Split the image into RGB channels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    r_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    b_channel = image[:, :, 2]

    # Calculate histograms for each channel
    r_hist, _ = np.histogram(r_channel, bins=np.arange(256))
    g_hist, _ = np.histogram(g_channel, bins=np.arange(256))
    b_hist, _ = np.histogram(b_channel, bins=np.arange(256))

    # Normalize histograms to get probabilities
    r_prob = r_hist / float(r_hist.sum())
    g_prob = g_hist / float(g_hist.sum())
    b_prob = b_hist / float(b_hist.sum())

    # Calculate information entropy for each channel
    r_entropy = -np.sum(r_prob * np.log2(r_prob + 1e-10))
    g_entropy = -np.sum(g_prob * np.log2(g_prob + 1e-10))
    b_entropy = -np.sum(b_prob * np.log2(b_prob + 1e-10))

    # Combine entropies from all channels
    total_entropy = (r_entropy + g_entropy + b_entropy) / 3.0

    return total_entropy



def main():
    original_img_path = f"img/sample/plain_sample/3_miniature poodle.bmp"

    original_img = cv2.imread(original_img_path)
    original_entropy = calculate_rgb_image_entropy(original_img)
    plot_histograms(original_img, f"original_img_entropy{original_entropy:.2f}")
    # print(f"RGB original Entropy ={original_entropy}")

    for sb_size in [2, 7, 14]:
        # print("\n")
        ind_img_path = f"img/sample/img_sb_{sb_size}/ind_img_{sb_size}.bmp"

        ind_img = cv2.imread(ind_img_path)
        ind_entropy = calculate_rgb_image_entropy(ind_img)
        print(f"RGB ind_mb14_{sb_size} Entropy ={ind_entropy}")
        plot_histograms(ind_img, f"ind_mb14_sb{sb_size}_entropy{ind_entropy:.2f}")

        com_img_path = f"img/sample/img_sb_{sb_size}/com_img_{sb_size}.bmp"
        com_img = cv2.imread(com_img_path)
        com_entropy = calculate_rgb_image_entropy(com_img)
        plot_histograms(com_img, f"com_mb14_sb{sb_size}_entropy{com_entropy:.2f}")

        print(f"RGB com_mb14_{sb_size} Entropy ={com_entropy}")


if __name__ == '__main__':
    main()
