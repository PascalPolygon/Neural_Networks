# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # House prices prediction with linear regression model

# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import csv


# %%
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
            else:
                data_arr.append(row)

            ln_cnt += 1

    ln_cnt -= 1  # Correction for counting the first line which contains no data
    return field_nb, ln_cnt, data_arr


# %%
def gen_plots(data):
    # print(data[:,0])
    x = data[:, 0]
    y = data[:, 1]

    area = np.pi*3

    # plt.scatter(x, y, s=area, alpha=0.5)
    fig = plt.figure(figsize=(10, 10))

    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # plt.xticks(np.arange(0, 1, step=0.2))
    # plt.yticks(np.arange(0, 1, step=0.2))
    axes.scatter(x, y, s=area, alpha=0.5)
    # plt.plot(x, y, 'o', alpha=0.5)

    # plt.show()
    plt.savefig("lotFrontage_1.png")
    plt.show()

    # print(len(x))
    # print(len(y))


# %%
dataFile = './dataset/housing_dataset.csv'
cols, rows, data_arr = read_csv_file(dataFile)
data_mtx = np.array(data_arr)
data_mtx.resize((rows, cols))
print(data_mtx.shape)

gen_plots(data_mtx)
# print(data_arr)
# print(data_arr[:,0])

# %%
