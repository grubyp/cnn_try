import numpy as np

def create_conv(size):
    return np.random.normal(0, 0.01, size)

def constant(size):
    return np.ones(size) * 0.1

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
    # relu
    result = np.maximum(result, 0)
    return result

def calc_maxpool(img, maxpool, maxpool_step):
    img_size = np.shape(img)
    img_num = img_size[0]
    img_row = img_size[1]
    img_col = img_size[2]
    img_deep = img_size[3]
    pool_row = maxpool[0]

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

def calc_fc(img, fc_w, fc_b):
    img_size = np.shape(img)
    fc_w_size = np.shape(fc_w)
    img_num = img_size[0]
    result = np.zeros([img_num, fc_w_size[1], 1])
    for i in range(img_num):
        result[i] = (np.matmul(fc_w.T, img[i]) + fc_b)
    # relu
    result = np.maximum(result, 0)
    return result

def softmax(img):
    img_size = np.shape(img)
    img_num = img_size[0]
    result = np.zeros(img_size)
    for i in range(img_num):
        # img[i] -= np.max(img[i])
        result[i] = np.exp(img[i]) / np.sum(np.exp(img[i]))
    return result

def bp_softmax(sm, lab_mat):
    loss = -1 * np.sum(lab_mat * np.log(sm), axis=1)
    cost = np.mean(loss)
    grad = sm - lab_mat
    print('cost = %.4f' % cost)
    return grad

def bp_fc(last_fc, fc_delta, fc_w):
    img_num = last_fc.shape[0]
    last_fc_size = fc_w.shape[0]
    fc_size = fc_w.shape[1]
    dw_sum = np.zeros((img_num, last_fc_size, fc_size))
    for i in range(img_num):
        dw_sum[i] = np.matmul(last_fc[i], fc_delta[i].T)
    delta_w = np.mean(dw_sum, axis=0)
    delta_b = np.mean(fc_delta, axis=0)
    fc_grad = {'dw': delta_w, 'db': delta_b}
    last_fc_delta = np.zeros((img_num, last_fc_size, 1))
    for i in range(img_num):
        last_fc_delta[i] = np.matmul(fc_w, fc_delta[i]) * (1-np.power(last_fc[i], 2))
    return fc_grad, last_fc_delta

def create_mask(x):
    mask = (x == np.max(x))
    return mask

def bp_pool(conv_prev, pool_delta, maxpool, maxpool_step):
    pool_delta_size = np.shape(pool_delta)
    pool_row = maxpool[0]
    result = np.zeros(conv_prev.shape)
    for n in range(pool_delta_size[0]):
        aconv_prev = conv_prev[n, :, :, :]
        for i in range(pool_delta_size[1]):
            for j in range(pool_delta_size[2]):
                for d in range(pool_delta_size[3]):
                    vert_start = i * maxpool_step
                    vert_end = vert_start + pool_row
                    horiz_start = j * maxpool_step
                    horiz_end = horiz_start + pool_row
                    mask = create_mask(aconv_prev[vert_start:vert_end, horiz_start:horiz_end, d])
                    result[n, vert_start:vert_end, horiz_start:horiz_end, d] += mask * pool_delta[n, i, j, d]
    return result

def bp_conv(conv_core, conv_delta):
    conv_delta_size = np.shape(conv_delta)
    conv_core_row = conv_core.shape[0]
    conv_core_deep = conv_core.shape[2]
    conv_core_step = 1
    conv_delta_prev = np.zeros((conv_delta_size[0], conv_delta_size[1], conv_delta_size[2], conv_core_deep))
    for n in range(conv_delta_size[0]):
        for i in range(conv_delta_size[1]):
            for j in range(conv_delta_size[2]):
                for d in range(conv_delta_size[3]):
                    vert_start = i * conv_core_step
                    vert_end = vert_start + conv_core_row
                    horiz_start = j * conv_core_step
                    horiz_end = horiz_start + conv_core_row


def train(train_img, train_lab, label_range):
    lab_size = np.shape(train_lab)
    lab_num = lab_size[0]
    lab_mat = np.zeros([lab_num, label_range, 1])
    for i in range(lab_num):
        lab_mat[i][train_lab[i]] = 1
    print('lab_mat.shape = ', lab_mat.shape)

    # 卷积层初始化
    conv_step = 1
    conv3_16_0 = create_conv((3, 3, 16))
    conv_b_1 = constant(16)
    conv3_16_1 = create_conv((3, 3, 16))
    conv_b_2 = constant(16)
    conv3_32_0 = create_conv((3, 3, 32))
    conv_b_3 = constant(32)
    conv3_32_1 = create_conv((3, 3, 32))
    conv_b_4 = constant(32)

    # 池化层初始化
    maxpool_step = 2
    maxpool = [2, 2]

    # 计算卷积
    # 第一层卷积
    conv_1 = calc_conv(train_img, conv3_16_0, conv_b_1, conv_step)
    print('conv_1.shape = ', conv_1.shape)
    # 第二层卷积
    conv_2 = calc_conv(conv_1, conv3_16_1, conv_b_2, conv_step)
    print('conv_2.shape = ', conv_2.shape)
    # 第一层池化
    pool_1 = calc_maxpool(conv_2, maxpool, maxpool_step)
    print('pool_1.shape = ', pool_1.shape)
    # 第三层卷积
    conv_3 = calc_conv(pool_1, conv3_32_0, conv_b_3, conv_step)
    print('conv_3.shape = ', conv_3.shape)
    # 第四层卷积
    conv_4 = calc_conv(conv_3, conv3_32_1, conv_b_4, conv_step)
    print('conv_4.shape = ', conv_4.shape)
    # 第二层池化
    pool_2 = calc_maxpool(conv_4, maxpool, maxpool_step)
    pool_2_size = np.shape(pool_2)
    print('pool_2_size.shape = ', pool_2_size)

    fc_0_size = pool_2_size[1] * pool_2_size[2] * pool_2_size[3]
    fc_w_0 = create_conv((fc_0_size, fc_0_size))
    fc_b_0 = constant((fc_0_size, 1))
    fc_1_size = label_range
    fc_w_1 = create_conv((fc_0_size, fc_1_size))
    fc_b_1 = constant((fc_1_size, 1))
    # 第一层全连接层
    fc_0 = np.reshape(pool_2, (pool_2_size[0], fc_0_size, 1))
    print('fc_0.shape = ', fc_0.shape)
    # 第二层全连接层
    fc_1 = calc_fc(fc_0, fc_w_0, fc_b_0)
    print('fc_1.shape = ', fc_1.shape)
    # 第三层全连接层
    fc_2 = calc_fc(fc_1, fc_w_1, fc_b_1)
    print('fc_2.shape = ', fc_2.shape)

    # softmax层
    sm = softmax(fc_2)
    print('sm.shape = ', sm.shape)

    # softmax层反馈
    fc_delta_2 = bp_softmax(sm, lab_mat)
    print('fc_delta_2.shape = ', fc_delta_2.shape)

    # 全连接层反馈
    fc_grad_1, fc_delta_1 = bp_fc(fc_1, fc_delta_2, fc_w_1)
    print('fc_delta_1.shape = ', fc_delta_1.shape)
    fc_grad_0, fc_delta_0 = bp_fc(fc_0, fc_delta_1, fc_w_0)
    print('fc_delta_0.shape = ', fc_delta_0.shape)

    # 第二层池化层反馈
    pool_delta_2 = np.reshape(fc_delta_0, pool_2_size)
    conv_delta_4 = bp_pool(conv_4, pool_delta_2, maxpool, maxpool_step)

    # 卷积层反馈
    bp_conv(conv3_32_1, conv_delta_4)

