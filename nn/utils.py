import struct
import numpy as np
import matplotlib.pyplot as plt

def load_mnist_old():
    train_file = '../data/train-images.idx3-ubyte'
    train_bin = open(train_file, 'rb')
    train_buf = train_bin.read()
    head = struct.unpack_from('>IIII', train_buf, 0)

    img_num = head[1]
    width = head[2]
    height = head[3]
    bits = img_num * width * height   # 60000 * 28 * 28
    bits_string = '>' + str(bits) + 'B'
    offset = struct.calcsize('>IIII')   # 偏移量，4个整数后
    imgs = struct.unpack_from(bits_string, train_buf, offset)
    train_bin.close()
    imgs = np.reshape(imgs, [img_num, width, height])
    plt.imshow(imgs[0])
    plt.show()

def load_mnist():
    train_lp = '../data/train-labels.idx1-ubyte'
    train_ip = '../data/train-images.idx3-ubyte'
    test_lp = '../data/t10k-labels.idx1-ubyte'
    test_ip = '../data/t10k-images.idx3-ubyte'
    with open(train_lp, 'rb') as train_label_path:
        magic, n = struct.unpack('>II', train_label_path.read(8))
        train_lab = np.fromfile(train_label_path, dtype=np.uint8)

    with open(train_ip, 'rb') as train_img_path:
        magic, num, rows, cols = struct.unpack('>IIII', train_img_path.read(16))
        train_img = np.fromfile(train_img_path, dtype=np.uint8).reshape(num, rows, cols, 1)

    with open(test_lp, 'rb') as test_label_path:
        magic, n = struct.unpack('>II', test_label_path.read(8))
        test_lab = np.fromfile(test_label_path, dtype=np.uint8)

    with open(test_ip, 'rb') as test_img_path:
        magic, num, rows, cols = struct.unpack('>IIII', test_img_path.read(16))
        test_img = np.fromfile(test_img_path, dtype=np.uint8).reshape(num, rows, cols, 1)

    return train_lab, train_img, test_lab, test_img

def look(conv):
    img = conv / np.max(conv) * 256
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    train_lab, train_img, test_lab, test_img = load_mnist()
    print(train_img.shape)
    print(test_img.shape)
