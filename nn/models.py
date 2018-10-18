import numpy as np

def create_conv(x, y, z):
    return np.random.randn(x, y, z)

def calc_conv(imgs, conv, conv_b, conv_step):
    img_size = np.shape(imgs)
    conv_size = np.shape(conv)

    # padding
    img_num = img_size[0]
    img_row = img_size[1]
    img_col = img_size[2]
    img_deep = img_size[3]
    conv_row = conv_size[0]
    conv_deep = conv_size[2]
    pad = conv_row - 1   # img_row - (img_row - conv_row + 1)
    img_padded = np.zeros([img_size[0], img_row + pad, img_col + pad, img_deep])
    img_padded[:, 1:-1, 1:-1, :] = imgs

    # 卷积
    result = np.zeros([img_num, img_row, img_col, conv_deep])
    for n in range(img_num):
        for i in range(img_row):
            for j in range(img_col):
                for k in range(conv_deep):
                    for d in range(img_deep):
                        result[n, i, j, k] = result[n, i, j, k] + np.sum(img_padded[n, i:i+conv_row, j:j+conv_row, d] * conv[:, :, k])
                    result[n, i, j, k] = result[n, i, j, k] + conv_b[k]
                    if(result[n, i, j, k] < 0):
                        result[n, i, j, k] = 0
    return result

def calc_maxpool(img, maxpool, maxpool_step):
    img_size = np.shape(img)
    pool_size = np.shape(maxpool)
    img_num = img_size[0]
    img_row = img_size[1]
    img_col = img_size[2]
    img_deep = img_size[3]
    pool_row = pool_size[0]

    # 池化
    result_row = np.int(img_row / maxpool_step)
    result_col = np.int(img_col / maxpool_step)
    result = np.zeros([img_num, result_row, result_col, img_deep])
    for n in range(img_num):
        for i in range(result_row):
            for j in range(result_col):
                for d in range(img_deep):
                    result[n, i, j, d] = np.max(img[n, i * maxpool_step:i * maxpool_step + pool_row, j * maxpool_step:j * maxpool_step + pool_row, d])
    return result

def calc_fc(img, fc_w, fc_b, flag = False):
    img_size = np.shape(img)
    fc_w_size = np.shape(fc_w)
    img_num = img_size[0]
    img_deep = img_size[1]
    if(flag):
        result = np.zeros([img_num, fc_w_size[0]])
        for i in range(img_num):
            result[i] = (np.matmul(fc_w, img[i]) + fc_b).T
    else:
        result = np.zeros([img_num, fc_w_size[0], 1])
        for i in range(img_num):
            result[i] = (np.matmul(fc_w, img[i]) + fc_b)
    return result

def calc_softmax(img):
    img_size = np.shape(img)
    img_num = img_size[0]
    img_deep = img_size[1]
    result = np.zeros([img_num, img_deep])
    for i in range(img_num):
        img[i] -= np.max(img[i])
        result[i] = np.exp(img[i]) / np.sum(np.exp(img[i]))
    return result

def train(train_img, train_lab, label_range):
    lab_size = np.shape(train_lab)
    lab_num = lab_size[0]
    lab_arr = np.zeros([lab_num, label_range])
    for i in range(lab_num):
        lab_arr[i][train_lab[i]] = 1
    print(lab_arr)

    # 卷积层初始化
    conv_step = 1
    conv3_16_0 = create_conv(3, 3, 16)
    conv_b_1 = np.zeros(16)
    conv3_16_1 = create_conv(3, 3, 16)
    conv_b_2 = np.zeros(16)
    conv3_32_0 = create_conv(3, 3, 32)
    conv_b_3 = np.zeros(32)
    conv3_32_1 = create_conv(3, 3, 32)
    conv_b_4 = np.zeros(32)

    # 池化层初始化
    maxpool_step = 2
    maxpool = [2, 2]

    # 计算卷积
    # 第一层卷积
    conv_1 = calc_conv(train_img, conv3_16_0, conv_b_1, conv_step)
    print(conv_1.shape)
    # 第二层卷积
    conv_2 = calc_conv(conv_1, conv3_16_1, conv_b_2, conv_step)
    print(conv_2.shape)
    # 第一层池化
    pool_1 = calc_maxpool(conv_2, maxpool, maxpool_step)
    print(pool_1.shape)
    # 第三层卷积
    conv_3 = calc_conv(pool_1, conv3_32_0, conv_b_3, conv_step)
    print(conv_3.shape)
    # 第四层卷积
    conv_4 = calc_conv(conv_3, conv3_32_1, conv_b_4, conv_step)
    print(conv_4.shape)
    # 第二层池化
    pool_2 = calc_maxpool(conv_4, maxpool, maxpool_step)
    pool_2_size = np.shape(pool_2)
    print(pool_2_size)

    fc_0_size = pool_2_size[1] * pool_2_size[2] * pool_2_size[3]
    fc_w_0 = np.random.randn(fc_0_size, fc_0_size)
    fc_b_0 = np.random.randn(fc_0_size, 1)
    fc_1_size = label_range
    fc_w_1 = np.random.randn(fc_1_size, fc_0_size)
    fc_b_1 = np.random.randn(fc_1_size, 1)
    # 第一层全连接层
    fc_0 = np.reshape(pool_2, [pool_2_size[0], fc_0_size, 1])
    print(fc_0.shape)
    # 第二层全连接层
    fc_1 = calc_fc(fc_0, fc_w_0, fc_b_0)
    print(fc_1.shape)
    # 第三层全连接层
    fc_2 = calc_fc(fc_1, fc_w_1, fc_b_1, flag=True)
    print(fc_2.shape)

    # softmax
    sm = calc_softmax(fc_2)
    print(sm.shape)