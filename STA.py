import numpy as np
from neuron import h, gui
import record_1comp as r1
import efel
from IVL_helper import *
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


"""IVL states 0 to 5. [ge0, gi0, stde, stdi]. tau given by Alex."""
# ivl_paras = np.array([[0.01, 0.03888889, 0.00083333, 0.00944444],
#                       [0.01, 0.03666667, 0.00094444, 0.00966667],
#                       [0.01, 0.03444444, 0.0005, 0.00933333],
#                       [0.01, 0.03222222, 0.00066667, 0.009],
#                       [0.01, 0.03111111, 0.00061111, 0.009]])

ivl_paras = np.array([[0.00321429, 0.01085714, 0.00068571, 0.00214286],
                     [0.00342857, 0.01028571, 0.00057143, 0.00214286],
                     [0.00342857, 0.01114286, 0., 0.00428571],
                     [0.00385714, 0.01171429, 0., 0.00428571],
                     [0.00428571, 0.01171429, 0.00045714, 0.00357143],
                     [0.00385714, 0.01142857, 0.00011429, 0.00571429],
                     [0.00428571, 0.01114286, 0., 0.00428571],
                     [0.00407143, 0.01, 0., 0.00428571],
                     [0.00364286, 0.00828571, 0., 0.00428571],
                     [0.0045, 0.01085714, 0.00011429, 0.00571429]])

tau_e = 12.7
tau_i = 12.05


def one_state_sta_run(paras, runtime, pretime, procnum=0, new_seeds=None, shared_dic=None,  savepath=None):
    """Return an 3 dimernsional array. Each entry is a two dimensional matrix of STA of a seed at fi"""
    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1
    OU_in.g_e0 = paras[0]
    OU_in.g_i0 = paras[1]
    OU_in.std_e = paras[2]
    OU_in.std_i = paras[3]
    OU_in.tau_e = tau_e
    OU_in.tau_i = tau_i

    h.ic_hold.delay = 0
    h.ic_hold.dur = runtime
    h.tstop = runtime

    ef_list = ["peak_indices"]
    efel.api.setThreshold(-30)
    efel.api.setDerivativeThreshold(1)
    t_vec = np.array((range(int(runtime / h.dt) + 1))) * h.dt
    vecs = r1.set_up_full_recording()
    pre_points = int(pretime / h.dt)
    count = 0
    final = []
    if new_seeds is None:
        new_seeds = seeds

    for i in range(new_seeds.shape[0]):
        OU_in.new_seed(new_seeds[i])
        h.run()
        v_vec = np.array(vecs[0])
        c_vecs = np.array([np.array(arr) for arr in vecs[1:]])
        trace = {'V': v_vec, 'T': t_vec, 'stim_start': [0], 'stim_end': [t_vec[-1]]}

        spike_idx = efel.getFeatureValues([trace], ef_list)[0]['peak_indices']
        for j in range(len(spike_idx)):
            curr_idx = spike_idx[j]
            slice_start = curr_idx - pre_points
            if j == 0:
                pre_surpass = slice_start < 0
            else:
                pre_surpass = spike_idx[j - 1] > slice_start
            if not pre_surpass:
                interspike_currents = c_vecs[:, slice_start:curr_idx]
                final.append(interspike_currents)
                count += 1
    if count == 0:
        return -1, -1
    else:
        final = np.array(final)
        if shared_dic is not None:
            shared_dic[procnum] = final
        if savepath is not None:
            np.save(savepath, final)
        return final


def one_state_sta_pert(paras, runtime, pretime, fi, procnum=0, new_seeds=None, shared_dic=None,  savepath=None, ext=False):
    """Return an 3 dimernsional array. Each entry is a two dimensional matrix of STA of a seed at fi"""
    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1
    OU_in.g_e0 = paras[0]
    OU_in.g_i0 = paras[1]
    OU_in.std_e = paras[2]
    OU_in.std_i = paras[3]
    OU_in.tau_e = tau_e
    OU_in.tau_i = tau_i

    h.ic_hold.delay = 0
    h.ic_hold.dur = runtime
    h.tstop = runtime

    # creating synapse
    stim = h.NetStim()  # average number of spikes. convert to ms, then s
    stim.noise = 0  # deterministic
    stim.start = 0

    syn = h.Exp2Syn(h.soma(0.5))
    syn.tau1 = 1.6  # ms rise time
    syn.tau2 = 12.0  # ms decay time
    syn.e = -87.1  # reversal potential

    netcon = h.NetCon(stim, syn)  # threshold is irrelevant with event-based source
    netcon.weight[0] = 0.0054

    if ext:
        syn = h.Exp2Syn(h.soma(0.5))
        syn.tau1 = 2.4  # ms rise time
        syn.tau2 = 12.7  # ms decay time
        syn.e = 0  # reversal potential

        netcon = h.NetCon(stim, syn)  # threshold is irrelevant with event-based source
        netcon.weight[0] = 0.006

    if fi == 0:
        stim.number = 0
        stim.interval = 1
    else:
        stim.interval = 1000 / fi  # interval in ms, but fi are in Hz
        stim.number = fi * (runtime / 1000)

    ef_list = ["peak_indices"]
    efel.api.setThreshold(-30)
    efel.api.setDerivativeThreshold(1)
    t_vec = np.array((range(int(runtime / h.dt) + 1))) * h.dt
    vecs = r1.set_up_full_recording()
    pre_points = int(pretime / h.dt)
    count = 0
    final = []
    if new_seeds is None:
        new_seeds = seeds

    for i in range(new_seeds.shape[0]):
        OU_in.new_seed(new_seeds[i])
        h.run()
        v_vec = np.array(vecs[0])
        all_vecs = np.array([np.array(arr) for arr in vecs])
        trace = {'V': v_vec, 'T': t_vec, 'stim_start': [0], 'stim_end': [t_vec[-1]]}

        spike_idx = efel.getFeatureValues([trace], ef_list)[0]['peak_indices']
        for j in range(len(spike_idx)):
            curr_idx = spike_idx[j]
            slice_start = curr_idx - pre_points
            if j == 0:
                pre_surpass = slice_start < 0
            else:
                pre_surpass = spike_idx[j - 1] > slice_start
            if not pre_surpass:
                interspike_iv = all_vecs[:, slice_start:curr_idx]
                final.append(interspike_iv)
                count += 1
    if count == 0:
        return -1, -1
    else:
        final = np.array(final)
        if shared_dic is not None:
            shared_dic[procnum] = final
        if savepath is not None:
            np.save(savepath, final)
        return final


def multi_seed_run(paras, runtime, pretime, fr_array, seed_to_fr, base_savepath, ext, text_file, numpts=None):
    shared_dic = mp.Manager().dict()
    proc_list = []
    for i in range(len(fr_array)):
        new_seeds = seed_to_fr[seed_to_fr[:, 1] == fr_array[i]][:, 0]
        p = mp.Process(target=one_state_sta_pert, args=(paras, runtime, pretime, fr_array[i],
                                                        fr_array[i], new_seeds, shared_dic, None, ext))
        p.start()
        proc_list.append(p)
    for p in proc_list:
        p.join()
    for i in range(len(fr_array)):
        try:
            stas = shared_dic[fr_array[i]][:numpts]
            avg = np.sum(stas, axis=0) / stas.shape[0]
            std = np.sum((stas - avg)**2, axis=0) / stas.shape[0]
            interim = np.array([avg, std])
            if numpts is not None:
                if ext:
                    np.save(base_savepath + f"_{fr_array[i]}_{numpts}s_ext.npy", interim)
                else:
                    np.save(base_savepath + f"_{fr_array[i]}_{numpts}s_inh.npy", interim)
            else:
                if ext:
                    np.save(base_savepath + f"_{fr_array[i]}_ext.npy", interim)
                else:
                    np.save(base_savepath + f"_{fr_array[i]}_inh.npy", interim)
            print(f'with fr {fr_array[i]}, there are {stas.shape[0]} {pretime} ms ISIs')
            text_file.write(f'with fr {fr_array[i]}, there are {stas.shape[0]} {pretime} ms ISIs\n')
        except KeyError:
            pass


def multi_seed_run_pertless(paras, runtime, pretime, fr_array, seed_to_fr, base_savepath):
    shared_dic = mp.Manager().dict()
    proc_list = []
    for i in range(len(fr_array)):
        new_seeds = seed_to_fr[seed_to_fr[:, 1] == fr_array[i]][:, 0]
        p = mp.Process(target=one_state_sta_run, args=(paras, runtime, pretime, fr_array[i], new_seeds, shared_dic, None))
        p.start()
        proc_list.append(p)
    for p in proc_list:
        p.join()
    for i in range(len(fr_array)):
        stas = shared_dic[fr_array[i]]
        avg = np.sum(stas, axis=0) / stas.shape[0]
        std = np.sum((stas - avg)**2, axis=0) / stas.shape[0]
        interim = np.array([avg, std])
        np.save(base_savepath + f"_{fr_array[i]}_pertless.npy", interim)


def seed_to_fr(bsr_path, seeds_array, fi_array, savepath):
    basline_ratio = np.load(bsr_path)
    results = np.zeros((seeds.shape[0], 2))
    for i in range(seeds_array.shape[0]):
        fr = fi_array[np.nanargmax(basline_ratio[i])]
        results[i][0] = seeds_array[i]
        results[i][1] = fr
    np.save(savepath, results)


def plot6(savepath, labels=None, load_path=None, avg_sta=None):
    if avg_sta is None:
        avg_sta = np.load(load_path)
    if labels is None:
        labels = ['IKa', 'IKdrf', 'Im', 'Il', 'INa', 'Ih']
    t_vec = np.array(range(1-avg_sta.shape[1], 1, 1)) * h.dt
    fig = plt.figure()
    for i in range(avg_sta.shape[0]):
        ax = fig.add_subplot(3, 2, i+1)
        ax.plot(t_vec, avg_sta[i], label=labels[i])
        plt.legend()
        plt.tight_layout()
    plt.savefig(savepath, dpi=250)


def plot6_percent(savepath, load_path, fig_name, labels=None, normalize=False):
    avg_std = np.load(load_path)
    if labels is None:
        labels = ['IKa', 'IKdrf', 'Im', 'Il', 'INa', 'Ih']
    t_vec = np.array(range(1 - avg_std.shape[2], 1, 1)) * h.dt
    fig = plt.figure()
    for i in range(avg_std.shape[1]):
        ax = fig.add_subplot(3, 2, i + 1)
        abs_total = np.sum(abs(avg_std[0]), axis=0)
        if not normalize:
            abs_total = 1
        ax.plot(t_vec, avg_std[0][i] / abs_total, label=labels[i])
        ax.plot(t_vec, (avg_std[0][i] - avg_std[1][i]) / abs_total, linestyle='--')
        ax.plot(t_vec, (avg_std[0][i] + avg_std[1][i]) / abs_total, linestyle='--')
        plt.legend()
        plt.tight_layout()
    plt.tight_layout()
    fig.suptitle(fig_name)
    plt.savefig(savepath)


if __name__ == "__main__":
    # bsr_dir = "data/spres_ps1_optw2_sl3"
    # save_dir = "data/spres_ps1_optw2_sl3"
    # num_states = 5

    bsr_dir = "data/spres_ps5"
    save_dir = "data/spres_ps5"
    num_states = 10
    seeds = np.array(range(5000)) + 2021
    fi_list = np.array([0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 23, 24, 25])
    # fr_array = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 23, 24, 25])
    # numpts = None
    # text_file = open("data/spres_ps5/num_isi", 'w')
    # fr_array = np.array([5, 9, 13, 25])
    # fr_array = np.array([2, 20])
    fr_array = np.array([0.5, 1, 2, 21, 23, 24])
    numpts = 50
    text_file = open("data/spres_ps5/num_isi_50s", 'w')

    for i in range(num_states):
        # seed_to_fr(bsr_path=f"{bsr_dir}/{i}/inh_baseline_ratio_vivo_single.npy", seeds_array=seeds, fi_array=fi_list,
        #            savepath=f"{save_dir}/{i}/seed_to_fr.npy")
        text_file.write(F"\n STATE {i}:\n")
        multi_seed_run(paras=ivl_paras[i],
                       runtime=10000,
                       pretime=200,
                       fr_array=fr_array,
                       seed_to_fr=np.load(f"{save_dir}/{i}/seed_to_fr.npy"),
                       base_savepath=f"{save_dir}/{i}/sta_at_fr/st{i}_fr",
                       ext=False, text_file=text_file, numpts=numpts)
    text_file.close()

