import numpy as np
from neuron import h, gui
import record_1comp as r1
import efel
from IVL_helper import *
import scipy.signal as signal
import matplotlib.pyplot as plt
import multiprocessing as mp
from currents_visualization import *

# Instantiate Model
h.load_file("init_1comp.hoc")
h.cvode_active(0)
h.dt = 0.1
h.steps_per_ms = 10
DNQX_Hold = 0.004
TTX_Hold = -0.0290514


def sub_thresh_calc(t_vec, v_vec):
    trace = {'V': v_vec, 'T': t_vec, 'stim_start': [0], 'stim_end': [t_vec[-1]]}
    ef_list = ["peak_indices"]

    efel.api.setThreshold(-30)
    efel.api.setDerivativeThreshold(1)

    features = efel.getFeatureValues([trace], ef_list)[0]

    v_spikeless = spike_ridder3(v_vec, features['peak_indices'], 70)

    v_mean = np.mean(v_spikeless)
    v_variance = np.var(v_spikeless)

    return v_mean, v_variance


def selector1(t_vec, v_vec):
    v_mean, v_variance = sub_thresh_calc(t_vec, v_vec)

    return -70 < v_mean < -65 and 7 < v_variance < 11


def selector2(t_vec, v_vec):
    """-70.5 < v_mean < -69.5; 8 < v_variance < 10. Examples from agm all fall within this range"""
    v_mean, v_variance = sub_thresh_calc(t_vec, v_vec)

    return -70 < v_mean < -69 and 8 < v_variance < 10


def selector3(t_vec, v_vec):
    v_mean, v_variance = sub_thresh_calc(t_vec, v_vec)

    return -70 < v_mean < -67.7 and 8 < v_variance < 10


def selector4(t_vec, v_vec):
    v_mean, v_variance = sub_thresh_calc(t_vec, v_vec)

    return -70 < v_mean < -69 and 7 < v_variance < 11


def single_select(paras, runtime, selector, shared_dic, proc_num):
    h.tstop = runtime

    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1
    OU_in.tau_e = 12.7
    OU_in.tau_i = 12.05
    t_vec = np.array((range(int(runtime / h.dt) + 1))) * h.dt
    v_hvec = r1.set_up_full_recording()[0]
    results = None
    for i in range(paras.shape[0]):
        OU_in.g_e0 = paras[i][0]
        OU_in.g_i0 = paras[i][1]
        OU_in.std_e = paras[i][2]
        OU_in.std_i = paras[i][3]
        h.run()
        v_vec = np.array(v_hvec)
        if selector(t_vec, v_vec):
            if results is None:
                results = paras[i]
            else:
                results = np.vstack((results, paras[i]))
    if not (results is None):
        shared_dic[proc_num] = results


def IVL_state_select(load_path, save_path, num_div, selector):
    paras = np.load(load_path)
    slice_size = int(paras.shape[0] // num_div)
    proc_list = []
    shared_dic = mp.Manager().dict()
    for i in range(0, paras.shape[0], slice_size):
        p = mp.Process(target=single_select, args=(paras[i: i+slice_size], 10000, selector, shared_dic, i))
        p.start()
        proc_list.append(p)
    for p in proc_list:
        p.join()

    results = None
    for key in list(shared_dic):
        if shared_dic[key] is None:
            sing_result = np.array([-1, -1, -1, -1])
        else:
            sing_result = shared_dic[key]

        if results is None:
            results = sing_result
        else:
            results = np.vstack((results, sing_result))

    if results is None:
        print("Mission failed, we will get'em next time.")
    else:
        np.save(save_path, results)


def state_stats(runtime, num_div, loadpath, savepath):
    paras = np.load(loadpath)
    slice_size = int(paras.shape[0] // num_div)
    proc_list = []
    shared_dic = mp.Manager().dict()
    for i in range(0, paras.shape[0], slice_size):
        p = mp.Process(target=state_stats_single, args=(paras[i: i + slice_size], runtime, shared_dic, i))
        p.start()
        proc_list.append(p)
    for p in proc_list:
        p.join()

    results = None
    for key in list(shared_dic):
        if shared_dic[key] is None:
            sing_result = np.array([-1, -1])
        else:
            sing_result = shared_dic[key]

        if results is None:
            results = sing_result
        else:
            results = np.vstack((results, sing_result))

    if results is None:
        print("Mission failed, we will get'em next time.")
    else:
        np.save(savepath, results)


def state_stats_single(paras, runtime, shared_dic, proc_num):
    h.tstop = runtime
    seeds = np.array(range(50)) + 2021
    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1
    OU_in.tau_e = 12.7
    OU_in.tau_i = 12.05
    t_vec = np.array((range(int(runtime / h.dt) + 1))) * h.dt
    v_hvec = r1.set_up_full_recording()[0]
    results = None

    ef_list = ["peak_indices"]
    efel.api.setThreshold(-30)
    efel.api.setDerivativeThreshold(1)

    for i in range(paras.shape[0]):
        OU_in.g_e0 = paras[i][0]
        OU_in.g_i0 = paras[i][1]
        OU_in.std_e = paras[i][2]
        OU_in.std_i = paras[i][3]
        interim = np.zeros(50)
        for j in range(50):
            OU_in.new_seed(seeds[j])
            h.run()
            v_vec = np.array(v_hvec)
            trace = {'V': v_vec, 'T': t_vec, 'stim_start': [0], 'stim_end': [t_vec[-1]]}
            features = efel.getFeatureValues([trace], ef_list)[0]
            spike_rate = len(features['peak_indices']) / t_vec[-1] * 1000
            interim[j] = spike_rate
        if results is None:
            results = np.array([np.min(interim), np.max(interim), paras[i][0], paras[i][1], paras[i][2], paras[i][3]])
        else:
            results = np.vstack((results, np.array([np.min(interim), np.max(interim), paras[i][0], paras[i][1],
                                                    paras[i][2], paras[i][3]])))
    if not (results is None):
        shared_dic[proc_num] = results


if __name__ == "__main__":
    # t_vec = np.load("agm_ex/IVL_tvec.npy")
    # v_vec = np.load("agm_ex/IVL_vvec_example0.npy")
    # print(sub_thresh_calc(t_vec, v_vec))
    # plt.plot(t_vec[:10000], v_vec[:10000])
    # plt.savefig("agm_ex/ex0.png")
    # plt.close()
    #
    # t_vec = np.load("agm_ex/IVL_tvec.npy")
    # v_vec = np.load("agm_ex/IVL_vvec_example1.npy")
    # print(sub_thresh_calc(t_vec, v_vec))
    # plt.plot(t_vec[:10000], v_vec[:10000])
    # plt.savefig("agm_ex/ex1.png")
    # plt.close()
    #
    # t_vec = np.load("agm_ex/IVL_tvec.npy")
    # v_vec = np.load("agm_ex/IVL_vvec_example2.npy")
    # print(sub_thresh_calc(t_vec, v_vec))
    # plt.plot(t_vec[:10000], v_vec[:10000])
    # plt.savefig("agm_ex/ex2.png")
    # plt.close()
    #
    # t_vec = np.load("agm_ex/IVL_tvec.npy")
    # v_vec = np.load("agm_ex/IVL_vvec_example3.npy")
    # print(sub_thresh_calc(t_vec, v_vec))
    # plt.plot(t_vec[:10000], v_vec[:10000])
    # plt.savefig("agm_ex/ex3.png")
    # plt.close()
    #
    # t_vec = np.load("agm_ex/IVL_tvec.npy")
    # v_vec = np.load("agm_ex/IVL_vvec_example4.npy")
    # print(sub_thresh_calc(t_vec, v_vec))
    # plt.plot(t_vec[:10000], v_vec[:10000])
    # plt.savefig("agm_ex/ex4.png")
    # plt.close()

    # IVL_state_select(load_path="data/IVL_search/full_search_4_agm.npy", save_path="data/IVL_select/fs4_agm_sl1.npy", 5)

    # IVL_state_select(load_path="full_search_2_agm.npy", save_path="fs2_agm_sl1.npy",
    #                  num_div=25, selector=selector1)

    # IVL_state_select(load_path="data/IVL_select/fs2_agm_sl1.npy", save_path="data/IVL_select/fs2_agm_sl2.npy",
    #                  num_div=4, selector=selector2)

    # IVL_state_select(load_path="data/IVL_select/fs2_agm_sl1.npy", save_path="data/IVL_select/fs2_agm_sl3.npy",
    #                  num_div=4, selector=selector3)

    # IVL_state_select(load_path="data/IVL_select/fs2_agm_sl1.npy", save_path="data/IVL_select/fs2_agm_sl4.npy",
    #                  num_div=4, selector=selector4)

    # IVL_state_select(load_path="data/IVL_select/ps1_optw2.npy", save_path="data/IVL_select/ps1_optw2_sl3.npy",
    #                  num_div=4, selector=selector3)
    #
    # state_stats(10000, num_div=5, loadpath="data/IVL_select/ps1_optw2_sl3.npy",
    #             savepath="data/IVL_select/ps1_optw2_sl3_stats.npy")

    IVL_state_select(load_path="data/IVL_select/ps2.npy", save_path="data/IVL_select/ps2_sl3.npy",
                     num_div=4, selector=selector3)

    # para = np.load("data/IVL_select/ps1_optw2_sl3_stats.npy")
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(para[:, 0], para[:, 1], s=1)
    # values = []
    # dtype = [('minr', float), ('maxr', float), ('ge', float), ('gi', float), ('stde', float), ('stdi', float)]
    # for i in range(para.shape[0]):
    #     values.append((para[i][0], para[i][1], para[i][2], para[i][3], para[i][4], para[i][5]))
    # para_sorted = np.array(values, dtype=dtype)
    # para_sorted = np.sort(para_sorted, order=['minr', 'maxr'])
    # para_sorted = para_sorted[[36, 143, 173, 181, 184]]
    # for i in range(para_sorted.shape[0]):
    #     ax.scatter(para_sorted[i][0], para_sorted[i][1], c='red')
    # ax.set_xlabel("min rate")
    # ax.set_ylabel("max rate")
    # plt.savefig("ps1_agm_sl3_stats_ex.png")
    # # plt.show()
    # plt.close()
    # print(para_sorted)
    # dic = mp.Manager().dict()
    # state_stats_single(np.array([para[53][2:]]), 10000, dic, 0)
    # print(dic[0][0], dic[0][1])
    # print(para[53][:2])
