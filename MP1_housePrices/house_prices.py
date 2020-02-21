import matplotlib.pyplot as plt
import numpy as np
import os
import csv


def get_X_and_Y(data):
    X = data[1:data.shape[0], 1:data.shape[1]]
    Y = data[1:data.shape[0], 0]

    Y = Y.astype(float)
    X = X.astype(float)

    X = augment(X)
    Y = np.c_[Y]

    return X, Y


def lin_reg_predict(data, W):
    print("dims data: "+str(data.shape))
    X, true_Y = get_X_and_Y(data)
    MSE_test = compute_MSE(true_Y, np.dot(X, W))
    print("MSE test: %f" % MSE_test)
    return MSE_test


def rmv_bin(data_mtx, start, len):
    cursor = start+len
    rows_to_rmv = range(cursor, cursor+len)
    new_mtx = np.delete(data_mtx, rows_to_rmv, 0)
    print("Cursor at: %d" % cursor)
    return new_mtx, cursor


def compute_bin_MSE(input, output, W):
    X = input.astype(float)
    Y = output.astype(float)
    X = augment(X)
    Y = np.c_[Y]
    return compute_MSE(Y, np.dot(X, W))


def get_bin(data_mtx, start, len):
    cursor = start+len
    bin = data_mtx[start:cursor, :]
    # print('Cursor: %d' % cursor)
    # print('Dims: '+ str(bin.shape))
    return bin


def gen_fld_sq(data_mtx, fld_to_sq):
    fields = data_mtx[0]
    idx = np.where(fields == fld_to_sq)
    year_blt = data_mtx[1:data_mtx.shape[0], idx].astype(float)
    year_blt_sqd = year_blt**2
    year_blt_sqd = year_blt_sqd.astype(str)
    year_blt_sqd = np.insert(year_blt_sqd, 0, fld_to_sq+"Sq")
    return year_blt_sqd


def rmv_col(mtx, fld_to_rmv):
    fields = mtx[0]
    idx = np.where(fields == fld_to_rmv)
    print("Deleting col %d" % idx)
    new_mtx = np.delete(mtx, idx, 1)

    return new_mtx


def save_csv_file(file_loc, data, fields):
    with open(file_loc, mode='w') as file:
        file_writer = csv.writer(file, delimiter=',', quotechar='"')
        file_writer.writerow(fields)  # write fields outside the loop
        for row in range(data.shape[0]):
            file_writer.writerow(data[row])


def generate_sets(data, max_trn):
    fields = data[0]

    trn_data = data[1:max_trn+1, :]
    test_data = data[max_trn+1:data.shape[0]]
    print("Training set: "+str(trn_data.shape))
    print("Test set: "+str(test_data.shape))

    trn_file_loc = '/home/pascal/Neural_Networks/MP1_housePrices/dataset/training_set.csv'
    test_file_loc = '/home/pascal/Neural_Networks/MP1_housePrices/dataset/testing_set.csv'

    save_csv_file(trn_file_loc, trn_data, fields)
    save_csv_file(test_file_loc, test_data, fields)


def augment(M):
    # lose 2-3 precision points through rounding (but oh well!)
    M = np.insert(M, 0, 1, axis=1)
    return M


def compute_MSE(Y, Y_hat):
    N = len(Y)
    MSE = (1/N)*((np.linalg.norm(Y-Y_hat))**2)
    # print('MSE train: %f' % MSE)
    return MSE


def grad_desc(Y, X, W, lrn_rate, epochs):
    N = len(Y)
    X_t = np.transpose(X)

    # print("Training (Please don't disturb) ...")
    for epoch in range(epochs):
        D_w = (2/N)*(np.dot(X_t, (np.dot(X, W)-Y)))
        W = W - np.dot(lrn_rate, D_w)
        compute_MSE(Y, np.dot(X, W))
     #     print("####################")
    return W


def lin_reg(X, Y):
    # initial guessing weights
    # The plus 1 to the rows is to include the intercept Wo
    W = np.zeros((X.shape[1]+1, 1))
    W = W.astype(float)
    # print(W)
    X = X.astype(float)
    Y = Y.astype(float)
    # Augment input matrix X (1's in col 0)
    X = augment(X)
    # print("New input dims")
    # print(X.shape)
    Y_hat = np.dot(X, W)

    Y = np.c_[Y]  # convert to column vector
    MSE = compute_MSE(Y, Y_hat)
    # Result of gradient descent is the weights
    W = grad_desc(Y, X, W, 0.01, 100000)

    return W


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

    data_mtx = np.array(data_arr)
    data_mtx.resize((ln_cnt, field_nb))

    return data_mtx


def gen_plots(data):
    # first row of data contains fields we will use for labels
    fields = data[0]
    data = data[1:data.shape[0], :]  # submatrix to ignore row 0
    cols = data.shape[1]
    print(cols)

    y = data[:, 0]

    for col_id in range(1, cols):
        x = data[:, col_id]
        print(col_id)

        area = np.pi*3
        plt.scatter(x, y, s=area, c='b', alpha=0.5)
        plt.title(fields[0]+" Vs "+fields[col_id])
        plt.xlabel(fields[col_id])
        plt.ylabel(fields[0])

        plt.savefig(
            "/home/pascal/Neural_Networks/MP1_housePrices/figures/plot_"+fields[col_id]+".png")
        # plt.show()


def generate_sets(data, max_trn):
    trn_set = data[1:max_trn, :]
    test_set = data[max_trn+1:data.shape[0]-1]
    print("Training set: "+str(trn_set.shape))
    print("Test set: "+str(test_set.shape))


# Set contains 1st 1000 rows for training
trn_file = '/home/pascal/Neural_Networks/MP1_housePrices/dataset/training_set.csv'
data_mtx = read_csv_file(trn_file)
print("Training set dims:")
print(data_mtx.shape)

# Uncomment below function call to remove a feature from data (OverallQuall in this case)
# data_mtx = rmv_col(data_mtx, "OverallQual")
# print("Dim without OverallQual: ", str(data_mtx.shape))

# Uncomment below function call to generate plots of features vs SalePrice
# gen_plots(data_mtx)

year_blt_sq = gen_fld_sq(data_mtx, "YearBuilt")

year_blt_sq = np.c_[year_blt_sq]

# add year built square to data
data_mtx = np.append(data_mtx, year_blt_sq, axis=1)
print(data_mtx.shape)

W_size = data_mtx.shape[1]-1  # -1 is to discard output in col 0
# print(bin_size)

W = np.zeros((int(W_size)+1, 1))
W_sum = np.zeros((int(W_size)+1, 1))
W = W.astype(float)
W_sum = W_sum.astype(float)
print(W.shape)
bin_epoch = 1

# rmv_bin(data_mtx, cursor, 239)
folds = 5
bin_size = int((data_mtx.shape[0]-1)/5)

print("bin size: %d" % bin_size)

cursor = 1

for i in range(5):

    ho_bin = get_bin(data_mtx, cursor, bin_size)
    trn_data, cursor = rmv_bin(data_mtx, cursor, bin_size)
    input = trn_data[1:trn_data.shape[0], 1:trn_data.shape[1]]
    output = trn_data[1:trn_data.shape[0], 0]
    test_output = ho_bin[:, 0]

    print("Training (Please don't disturb) ...")
    W_i = lin_reg(input, output)
    W_sum += W_i
    W = W_sum/bin_epoch

    MSE_trn = compute_bin_MSE(input, output, W)

    print("Fold %d MSE_train = %f " % (bin_epoch, MSE_trn))

    bin_epoch += 1

print("Training complete! Final MSE train: %f" % MSE_trn)

# Set contains 1st 1000 rows for training
# test_file = './dataset/testing_set.csv'
test_file = '/home/pascal/Neural_Networks/MP1_housePrices/dataset/testing_set.csv'
test_data_mtx = read_csv_file(test_file)

test_year_blt_sq = gen_fld_sq(test_data_mtx, "YearBuilt")
test_year_blt_sq = np.c_[test_year_blt_sq]  # Covert to column vect
# add year built square to data
test_data_mtx = np.append(test_data_mtx, test_year_blt_sq, axis=1)

lin_reg_predict(test_data_mtx, W)

res = data_mtx[1:data_mtx.shape[0], 0].astype(float)
X = data_mtx[1:data_mtx.shape[0], 1:data_mtx.shape[1]].astype(float)
X = augment(X)
preds = np.dot(X, W)
area = 5
lotFrt = data_mtx[1:data_mtx.shape[0], 3].astype(float)

fig, ax = plt.subplots()
ax.scatter(lotFrt, res, s=area, color='r',
           marker='o', alpha=0.5, label="Residuals")
ax.scatter(lotFrt, preds, s=area, color='c',
           marker='o', alpha=0.5, label="Fitted")
ax.legend()

plt.title("OverallQual Vs. SalePrice")
plt.xlabel("OverallQual")
plt.ylabel("SalePrice")
plt.savefig(
    "/home/pascal/Neural_Networks/MP1_housePrices/figures/results/Overqual_vs_SalePrice.png")
plt.show()
# plt.plot(preds, res, 'o', color='black');
plt.title("Fitted residual")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.scatter(preds, res, s=area, marker='o', alpha=0.5)
plt.savefig(
    "/home/pascal/Neural_Networks/MP1_housePrices/figures/results/fittedResidual.png")
