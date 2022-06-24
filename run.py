import numpy
from neuron import h, gui
import record_1comp as r1
import efel
import matplotlib.pyplot as plt
import multiprocessing as mp
# import efel
# import glob
# import IPython, os

from currents_visualization import *

### Instantiate Model ###
h.load_file("init_1comp.hoc")
h.cvode_active(0)
h.dt = 0.1
h.steps_per_ms = 10
DNQX_Hold = 0.004
TTX_Hold = -0.0290514
# cell = h.cell


def func0(dic):
    v_vec = r1.set_up_full_recording()[0]
    h.run()
    dic[0] = np.array(v_vec)


def func1(dic):
    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1
    # line = ivl_paras
    OU_in.E_i = -87.1
    OU_in.g_e0 = 0.01
    OU_in.g_i0 = 0.03
    OU_in.tau_e = 12.7
    OU_in.tau_i = 12.05
    v_vec = r1.set_up_full_recording()[0]

    OU_in.new_seed(2021)
    h.run()
    dic[1] = np.array(v_vec)


if __name__ == "__main__":
    mp.freeze_support()
    share_dic = mp.Manager().dict()
    p1 = mp.Process(target=func0, args=(share_dic,))
    p2 = mp.Process(target=func1, args=(share_dic,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    np.save("test0.npy", share_dic[0])
    np.save("test1.npy", share_dic[1])

    # plt.plot(np.load("test0.npy"))
    # plt.show()
    # plt.close()
    #
    # plt.plot(np.load("test1.npy"))
    # plt.show()
    # plt.close()
