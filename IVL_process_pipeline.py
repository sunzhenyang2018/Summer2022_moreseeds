import IVL_select as select
import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
# full process with selector 3


def full_process(dataname, datadir, num_div):
    select.IVL_state_select(load_path=f"{datadir}{dataname}", save_path=f"{datadir}{dataname[:-4]}_sl3.npy",
                            num_div=num_div, selector=select.selector3)
    select.state_stats(10000, num_div=num_div, loadpath=f"{datadir}{dataname[:-4]}_sl3.npy",
                       savepath=f"{datadir}{dataname[:-4]}_sl3_stats.npy")
    _ivl_rate_plotter(dataname, datadir)


def _ivl_rate_plotter(dataname, datadir):
    para = np.load(f"{datadir}{dataname[:-4]}_sl3_stats.npy")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(para[:, 0], para[:, 1], s=1)
    values = []
    dtype = [('minr', float), ('maxr', float), ('ge', float), ('gi', float), ('stde', float), ('stdi', float)]
    for i in range(para.shape[0]):
        values.append((para[i][0], para[i][1], para[i][2], para[i][3], para[i][4], para[i][5]))
    para_sorted = np.array(values, dtype=dtype)
    para_sorted = np.sort(para_sorted, order=['minr', 'maxr'])
    para_sorted = para_sorted[[10, 40, 70, 120, 170, 220, 260, 300, 330, 350]]
    for i in range(para_sorted.shape[0]):
        ax.scatter(para_sorted[i][0], para_sorted[i][1], c='red')
    ax.set_xlabel("min rate")
    ax.set_ylabel("max rate")
    plt.savefig(f"{dataname[:-4]}_sl3_stats_ex.png")


if __name__ == "__main__":
    # full_process(dataname="ps2.npy", datadir="data/IVL_select/", num_div=4)
    # full_process(dataname="ofs1.npy", datadir="data/IVL_select/", num_div=128)
    # ofs2_100b_gstdbr = np.load("data/IVL_search/ofs2_100b_gstdbr.npy")
    # ofs2_100b = ofs2_100b_gstdbr[:, 1:5]
    # np.save("data/IVL_select/ofs2_100b.npy", ofs2_100b)
    # full_process(dataname="ofs2_100b.npy", datadir="data/IVL_select/", num_div=128)
    # select.state_stats(10000, num_div=1, loadpath="data/IVL_select/ofs2_100b_sl3.npy",
    #                    savepath="data/IVL_select/ofs2_100b_sl3_stats.npy")
    # _ivl_rate_plotter(dataname="ofs2_100b.npy", datadir="data/IVL_select/")
    # full_process(dataname="ps3.npy", datadir="data/IVL_select/", num_div=4)
    # full_process(dataname="ps4.npy", datadir="data/IVL_select/", num_div=4)
    # full_process(dataname="ps5.npy", datadir="data/IVL_select/", num_div=4)
    _ivl_rate_plotter(dataname="ps5.npy", datadir="data/IVL_select/")
