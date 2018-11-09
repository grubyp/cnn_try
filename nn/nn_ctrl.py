from nn import utils, models
import time

def main():
    train_lab, train_img, test_lab, test_img = utils.load_mnist()

    start_time = time.time()

    train_num = 10
    models.train(train_img[0:train_num], train_lab[0:train_num], 10, iter=1000)

    end_time = time.time()
    print(end_time - start_time)

if __name__ == '__main__':
    main()
