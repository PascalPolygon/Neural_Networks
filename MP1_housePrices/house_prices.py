import matplotlib.pyplot as plt
import numpy as np
import os
import csv


def read_csv_file(file):
    data_arr = []
    with open(file, mode='r') as data:
        # data_reader = csv.DictReader(data) # to read as dictionary
        data_reader = csv.reader(data)
        ln_cnt = 0  # line count
        print(data_reader)
        for row in data_reader:
            if ln_cnt == 0:
                # Length of the first row which tells me how many fields there are in the file
                field_nb = len(row)
                data_arr.append(row)
            else:
                data_arr.append(row)

            ln_cnt += 1

    # ln_cnt -= 1 # Correction for counting the first line which contains no data
    return field_nb, ln_cnt, data_arr


def gen_plots(data):
    # first row of data contains fields we will use for labels
    fields = data[0]
    data = data[1:data.shape[0], :]  # submatrix to ignore row 0
    # print(data[0])
    # print(data[data.shape[0]-1])
    cols = data.shape[1]
    print(cols)

    y = data[:, 0]

    for col_id in range(1, cols):
        x = data[:, col_id]
        print(col_id)

        area = np.pi*3
        fig = plt.figure(figsize=(10, 10))

        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        axes.scatter(x, y, s=area, alpha=0.5)

        plt.title(fields[0]+" Vs "+fields[col_id])
        plt.xlabel(fields[col_id])
        plt.ylabel(fields[0])

        plt.savefig("plot_"+fields[col_id]+".png")


def augment(M):
    # lose 2-3 precision points through rounding (but oh well!)
    M = np.insert(M, 0, 1, axis=1)
    return M


def compute_MSE(Y, Y_hat):
    N = len(Y)
    MSE = (1/N)*((np.linalg.norm(Y-Y_hat))**2)
    print('MSE train: %f' % MSE)
    return MSE


def grad_desc(Y, X, W, lrn_rate, epochs):
    N = len(Y)
    X_t = np.transpose(X)

    print("Training (Please don't disturb) ...")

    for epoch in range(epochs):
        D_w = (2/N)*(np.dot(X_t, (np.dot(X, W)-Y)))
        W = W - np.dot(lrn_rate, D_w)
        # Y_hat = np.dot(X,D_w)
        print("Epoch: %d" % epoch)
        compute_MSE(Y, np.dot(X, W))
        print("####################")

    print("Training complete!")


def lin_reg(X, Y):
    # initial guessing weights
    # The plus 1 to the rows is to include the intercept Wo
    W = np.zeros((X.shape[1]+1, 1))
    # print(W)
    X = X.astype(float)
    Y = Y.astype(float)
    # Augment input matrix X (1's in col 0)
    X = augment(X)
    print("New input dims")
    print(X.shape)

    Y_hat = np.dot(X, W)

    Y = np.c_[Y]  # convert Y to column vector
    MSE = compute_MSE(Y, Y_hat)
    grad_desc(Y, X, W, 0.01, 10000)


dataFile = './dataset/housing_dataset.csv'
cols, rows, data_arr = read_csv_file(dataFile)
data_mtx = np.array(data_arr)
data_mtx.resize((rows, cols))
print(data_mtx.shape)

# gen_plots(data_mtx)

# sub matrix of input data (discards fields at the top)
input = data_mtx[1:data_mtx.shape[0], 1:data_mtx.shape[1]]
# row 1 to num of rows (discard fields at the top)
output = data_mtx[1:data_mtx.shape[0], 0]
del data_mtx  # We no longer need this, clean up space in memory

# build linear regression model
lin_reg(input, output)
