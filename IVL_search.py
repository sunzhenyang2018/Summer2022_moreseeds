import numpy as np
from neuron import h, gui
import record_1comp as r1
import efel
from IVL_helper import *
import scipy.signal as signal
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import multiprocessing as mp
from currents_visualization import *
import IVL_select

### Instantiate Model ###
h.load_file("init_1comp.hoc")
h.cvode_active(0)
h.dt = 0.1
h.steps_per_ms = 10
DNQX_Hold = 0.004
TTX_Hold = -0.0290514
# cell = h.cell


def testrun():
    h.tstop = 1000
    OU_in = h.Gfluct2(h.soma(0.5))
    vecs = r1.set_up_full_recording()
    h.run()

    vecs = [np.array(i) for i in vecs]

    plotCurrentscape_6_current(vecs[0], vecs[1:])
    plt.savefig("1_comp_plots/IVL/test.png")


def metric_testrun():
    h.tstop = 10000
    OU_in = h.Gfluct2(h.soma(0.5))
    vecs = r1.set_up_full_recording()
    h.run()
    v_vec = np.array(vecs[0])
    t_vec = np.array(range(len(v_vec))) * h.dt
    trace = {'V': v_vec, 'T': t_vec, 'stim_start': [0], 'stim_end': [10000]}
    ef_list = ["AP_begin_indices", "AP_end_indices", "ISI_CV", "peak_voltage", "mean_frequency"]

    features = efel.getFeatureValues([trace], ef_list)[0]
    spike_starts = features['AP_begin_indices']
    spike_ends = features['AP_end_indices']
    isicv = features["ISI_CV"][0]
    peakvm = features['peak_voltage']
    f = features['mean_frequency'][0]

    ap_amps = peakvm - v_vec[spike_starts]

    for i in range(len(spike_starts)):
        v_vec = np.hstack((v_vec[:spike_starts[i]], v_vec[spike_ends[i]+1:]))
        off_set = spike_ends[i] - spike_starts[i] + 1
        spike_starts -= off_set
        spike_ends -= off_set

    v_mean = np.mean(v_vec)
    v_variance = np.sum((v_vec - v_mean)**2) / len(v_vec)
    peakvm_avg = np.mean(ap_amps)

    print(int(v_mean > -70.588))
    print(int(v_variance > 2.2))
    print(int(isicv > 0.8))
    print(int(3 < f < 25))
    print(int(peakvm_avg < 40))

    score = (int(v_mean > -70.588)) + (int(v_variance > 2.2)) + (int(isicv > 0.8)) + (int(3 < f < 25)) - 5 * (int(peakvm_avg < 40))
    return score


def para_explorations():
    h.tstop = 1000
    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1
    line = [0.00519, 0.013, 0.0006, 0.00454]
    # line = np.load("data/IVL_select/fs2_agm_sl3.npy")[4]
    # line = line[line[:, 0] == 270]
    # line = line[int(line.shape[0]//2)]
    # line = line[1:]
    OU_in.g_e0 = line[0]
    OU_in.g_i0 = line[1]
    OU_in.std_e = line[2]
    OU_in.std_i = line[3]
    OU_in.tau_e = 12.7
    OU_in.tau_i = 12.05
    vecs = r1.set_up_full_recording()
    # OU_in.new_seed(2024)
    h.run()
    v_vec = np.array(vecs[0])
    t_vec = np.array(range(len(v_vec))) * h.dt
    print(evaluator3(t_vec, v_vec, printout=True))
    print(IVL_select.selector3(t_vec, v_vec))
    print(line[0], line[1], line[2], line[3])
    print(IVL_select.sub_thresh_calc(t_vec, v_vec))
    plt.plot(v_vec)
    plt.savefig("1_comp_plots/IVL/spike_res/explore.png")
    # trace = {'V': v_vec, 'T': t_vec, 'stim_start': [0], 'stim_end': [t_vec[-1]]}
    # ef_list = ["AP_begin_indices", "AP_end_indices", "ISI_CV", "peak_voltage", "mean_frequency"]
    #
    # features = efel.getFeatureValues([trace], ef_list)[0]
    # spike_starts = features['AP_begin_indices']
    # spike_ends = features['AP_end_indices']
    # isicv = features["ISI_CV"][0]
    # peakvm = features['peak_voltage']
    # f = features['mean_frequency'][0]
    #
    # ef_list = ["peak_indices"]
    # spike_indices = efel.getFeatureValues([trace], ef_list)[0]["peak_indices"]
    #
    # plt.plot(v_vec)
    # plt.savefig("1_comp_plots/IVL/test4.png")
    # plt.close()
    # v_vec = spike_ridder3(v_vec, spike_indices, 50)
    # plt.plot(v_vec)
    # plt.savefig("1_comp_plots/IVL/test5.png")
    # plt.close()
    # v_vec = v_vec[v_vec != 0]
    # plt.plot(v_vec)
    # plt.savefig("1_comp_plots/IVL/test6.png")
    # plt.close()


def para_explore_conductance(max_value, steps, run_time, savepath):
    values = np.linspace(0, max_value, steps)

    results = np.zeros((steps, steps))

    h.tstop = run_time
    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1
    OU_in.tau_e = 12.7
    OU_in.tau_i = 12.05
    vecs = r1.set_up_full_recording()

    t_vec = np.array((range(int(run_time/h.dt) + 1))) * h.dt
    for i in range(steps):
        for j in range(steps):
            # excitatory changes vertically, inhibitory changes horizontally
            OU_in.g_e0 = values[i]
            OU_in.g_i0 = values[j]
            h.run()
            v_vec = np.array(vecs[0])
            results[i][j] = evaluator3(t_vec, v_vec)

    np.save(savepath, results)


def para_explore_noise(max_value, steps, run_time, savepath):
    values = np.linspace(0, max_value, steps)

    results = np.zeros((steps, steps))

    h.tstop = run_time
    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1
    OU_in.g_e0 = 0.01
    OU_in.g_i0 = 0.03
    OU_in.tau_e = 12.7
    OU_in.tau_i = 12.05
    vecs = r1.set_up_full_recording()

    t_vec = np.array((range(int(run_time/h.dt) + 1))) * h.dt
    for i in range(steps):
        for j in range(steps):
            # excitatory changes vertically, inhibitory changes horizontally
            OU_in.std_e = values[i]
            OU_in.std_i = values[j]
            h.run()
            v_vec = np.array(vecs[0])
            results[i][j] = evaluator3(t_vec, v_vec)

    np.save(savepath, results)


def para_explore_tau(max_value, steps, run_time, savepath):
    values = np.linspace(0, max_value, steps)

    results = np.zeros((steps, steps))

    h.tstop = run_time
    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1
    OU_in.g_e0 = 0.01
    OU_in.g_i0 = 0.03
    vecs = r1.set_up_full_recording()

    t_vec = np.array((range(int(run_time/h.dt) + 1))) * h.dt
    for i in range(steps):
        for j in range(steps):
            # excitatory changes vertically, inhibitory changes horizontally
            OU_in.tau_e = values[i]
            OU_in.tau_i = values[j]
            h.run()
            v_vec = np.array(vecs[0])
            results[i][j] = evaluator3(t_vec, v_vec)

    np.save(savepath, results)


def score_plotter(max_value, steps, labels, score_path, savepath):
    """The score retreived should be a square matrix with maximum value of 4"""
    scores = np.load(score_path)
    fig = plt.figure()
    score_plot = fig.add_subplot(1, 15, (1,14))
    color_bar = fig.add_subplot(1, 15, 15)
    cmap = plt.cm.viridis
    norm = matplotlib.colors.Normalize(vmin=np.amin(scores), vmax=np.amax(scores))

    score_plot.imshow(scores, cmap=cmap)
    cb1 = matplotlib.colorbar.ColorbarBase(color_bar, cmap=cmap, norm=norm, orientation='vertical')

    tick_freq = range(0, steps, int(steps/10))
    ticks = np.round(np.linspace(0, max_value, steps)[tick_freq], 2).astype(str)

    score_plot.set_xlabel(labels[0])
    score_plot.set_xticks(tick_freq)
    score_plot.set_xticklabels(ticks, rotation=45)

    score_plot.set_ylabel(labels[1])
    score_plot.set_yticks(tick_freq)
    score_plot.set_yticklabels(ticks, rotation=45)
    cb1.set_label('Scores')

    plt.tight_layout()

    plt.savefig(savepath)


def ap_width_at_neg20mV():
    """To determine the time needed to be cut off around the -20 threshold"""
    run_time = 10000
    h.tstop = run_time
    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1
    OU_in.g_e0 = 0.01
    OU_in.g_i0 = 0.03
    OU_in.tau_e = 12.7
    OU_in.tau_i = 12.05
    v_hvec = r1.set_up_full_recording()[0]

    apc = h.APCount(h.soma(0.5))
    apc.thresh = -20
    spike_timing = h.Vector()
    apc.record(spike_timing)

    h.run()

    v_vec = np.array(v_hvec)
    v_vec[v_vec < -20] = 0
    v_vec[v_vec != 0] = 1

    spike_num = len(np.array(spike_timing))

    return np.sum(v_vec) / spike_num


def cut_away_tests(value_array):
    h.tstop = 1000
    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1
    OU_in.g_e0 = 0.01
    OU_in.g_i0 = 0.03
    OU_in.tau_e = 12.7
    OU_in.tau_i = 12.05
    vecs = r1.set_up_full_recording()
    h.run()
    v_vec = np.array(vecs[0])
    t_vec = np.array(range(len(v_vec))) * h.dt

    trace = {'V': v_vec, 'T': t_vec, 'stim_start': [0], 'stim_end': [t_vec[-1]]}
    ef_list = ["peak_indices"]
    spike_indices = efel.getFeatureValues([trace], ef_list)[0]["peak_indices"]

    for value in value_array:
        spikeless = spike_ridder3(v_vec, spike_indices, value)
        spikeless = spikeless[spikeless != 0]
        plt.plot(spikeless)
        plt.savefig("1_comp_plots/IVL/spike_cut_away/" + str(value) + ".png")
        plt.close()


def g_value_select(max_value, steps, run_time, savepath):
    values = np.linspace(0, max_value, steps)

    h.tstop = run_time
    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1

    apc = h.APCount(h.soma(0.5))
    apc.thresh = -20
    spike_timing = h.Vector()
    apc.record(spike_timing)

    vecs = r1.set_up_full_recording()
    t_vec = np.array((range(int(run_time/h.dt) + 1))) * h.dt

    results = None
    for i in range(steps):
        for j in range(steps):
            # excitatory changes vertically, inhibitory changes horizontally
            OU_in.g_e0 = values[i]
            OU_in.g_i0 = values[j]
            h.run()

            v_vec = np.array(vecs[0])
            score = evaluator3(t_vec, v_vec)
            rate = len(spike_timing) / run_time * 1000

            if score == 4:
                if results is None:
                    results = np.array([values[i], values[j], rate])
                else:
                    results = np.vstack((results, np.array([values[i], values[j], rate])))

    np.save(savepath, results)

    print(f"{min(results[:, 2])} <= rates <= {max(results[:, 2])}")
    print(f"total {results.shape[0]} rates gathered")


def g_value_select_aux(min_value, max_value, steps, run_time, savepath):
    values = np.linspace(min_value, max_value, steps)

    h.tstop = run_time
    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1

    apc = h.APCount(h.soma(0.5))
    apc.thresh = -20
    spike_timing = h.Vector()
    apc.record(spike_timing)

    vecs = r1.set_up_full_recording()
    t_vec = np.array((range(int(run_time/h.dt) + 1))) * h.dt

    results = None
    for i in range(steps):
        for j in range(steps):
            # excitatory changes vertically, inhibitory changes horizontally
            OU_in.g_e0 = values[i]
            OU_in.g_i0 = values[j]
            h.run()

            v_vec = np.array(vecs[0])
            score = evaluator3(t_vec, v_vec)
            rate = len(spike_timing) / run_time * 1000

            if score == 4:
                if results is None:
                    results = np.array([values[i], values[j], rate])
                else:
                    results = np.vstack((results, np.array([values[i], values[j], rate])))

    if results is None:
        results = np.array([-1, -1, -1])

    np.save(savepath, results)


def block_g_eval(ge0, gi0, run_time, proc_num, shared_dict, dest):
    h.tstop = run_time
    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1

    OU_in.std_e = 0.004
    OU_in.std_i = 0.008

    if not dest:
        OU_in.tau_e = 12.7
        OU_in.tau_i = 12.05

    apc = h.APCount(h.soma(0.5))
    apc.thresh = -20
    spike_timing = h.Vector()
    apc.record(spike_timing)

    vecs = r1.set_up_full_recording()
    t_vec = np.array((range(int(run_time/h.dt) + 1))) * h.dt
    results = None
    for i in range(ge0.shape[0]):
        for j in range(gi0.shape[0]):
            OU_in.g_e0 = ge0[i]
            OU_in.g_i0 = gi0[j]
            h.run()
            v_vec = np.array(vecs[0])
            score = evaluator3(t_vec, v_vec)
            rate = len(spike_timing) / run_time * 1000

            if score == 4:
                if results is None:
                    results = np.array([ge0[i], gi0[j], rate])
                else:
                    results = np.vstack((results, np.array([ge0[i], gi0[j], rate])))
    if results is None:
        results = np.array([-1, -1, -1])
    shared_dict[proc_num] = results


def std_value_select(max_value, steps, run_time, lines, savepath):
    """r is an array: [g_e0, g_i0]"""
    values = np.linspace(0, max_value, steps)

    h.tstop = run_time
    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1

    apc = h.APCount(h.soma(0.5))
    apc.thresh = -20
    spike_timing = h.Vector()
    apc.record(spike_timing)

    vecs = r1.set_up_full_recording()
    t_vec = np.array((range(int(run_time/h.dt) + 1))) * h.dt

    results = None

    for row_num in range(lines.shape[0]):
        r = lines[row_num]
        OU_in.g_e0 = r[0]
        OU_in.g_i0 = r[1]
        for i in range(steps):
            for j in range(steps):
                OU_in.std_e = values[i]
                OU_in.std_i = values[j]
                h.run()
                v_vec = np.array(vecs[0])
                score = evaluator3(t_vec, v_vec)

                if score == 4:
                    rate = len(spike_timing) / run_time * 1000

                    if results is None:
                        results = np.array([r[0], r[1], values[i], values[j], rate])
                    else:
                        results = np.vstack((results, np.array([r[0], r[1], values[i], values[j], rate])))
                j += 1
            i += 1
    if results is None:
        results = np.array([-1, -1, -1, -1, -1])

    np.save(savepath, results)
    # print(f"total {results.shape[0]} std pairs gathered")


def single_std_select(max_value, steps, run_time, lines, shared_dic, proc_num, dest):
    values = np.linspace(0, max_value, steps)

    h.tstop = run_time
    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1

    if not dest:
        OU_in.tau_e = 12.7
        OU_in.tau_i = 12.05

    apc = h.APCount(h.soma(0.5))
    apc.thresh = -30
    spike_timing = h.Vector()
    apc.record(spike_timing)

    vecs = r1.set_up_full_recording()
    t_vec = np.array((range(int(run_time / h.dt) + 1))) * h.dt

    results = None

    for row_num in range(lines.shape[0]):
        r = lines[row_num]
        OU_in.g_e0 = r[0]
        OU_in.g_i0 = r[1]
        for i in range(steps):
            for j in range(steps):
                OU_in.std_e = values[i]
                OU_in.std_i = values[j]
                h.run()
                v_vec = np.array(vecs[0])
                score = evaluator3(t_vec, v_vec)

                if score == 4:
                    rate = len(spike_timing) / run_time * 1000

                    if results is None:
                        results = np.array([r[0], r[1], values[i], values[j], rate])
                    else:
                        results = np.vstack((results, np.array([r[0], r[1], values[i], values[j], rate])))

    if results is None:
        results = np.array([-1, -1, -1, -1, -1])

    shared_dic[proc_num] = results


def tau_value_select(min_value, max_value, steps, run_time, load_path, save_path):
    rates = np.load(load_path)
    values = np.linspace(min_value, max_value, steps)

    h.tstop = run_time
    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1

    apc = h.APCount(h.soma(0.5))
    apc.thresh = -20
    spike_timing = h.Vector()
    apc.record(spike_timing)

    vecs = r1.set_up_full_recording()
    t_vec = np.array((range(int(run_time / h.dt) + 1))) * h.dt

    results = None
    min_rate = 25
    max_rate = 3

    for r in rates:
        OU_in.g_e0 = r[0]
        OU_in.g_i0 = r[1]
        OU_in.std_e = r[2]
        OU_in.std_i = r[3]
        i = 0
        count = 0
        while i < steps and count < 5:
            j = 0
            while j < steps and count < 5:
                OU_in.tau_e = values[i]
                OU_in.tau_i = values[j]
                h.run()
                v_vec = np.array(vecs[0])
                score = evaluator4(t_vec, v_vec)

                if score == 4:
                    count += 1
                    rate = len(spike_timing) / run_time * 1000
                    if rate > max_rate:
                        max_rate = rate
                    elif rate < min_rate:
                        min_rate = rate

                    if results is None:
                        results = np.array([r[0], r[1], r[2], r[3], values[i], values[j], rate])
                    else:
                        results = np.vstack((results, np.array([r[0], r[1], r[2], r[3], values[i], values[j], rate])))
                j += 1
            i += 1

    if results is None:
        np.save(save_path, np.array([0, 0, 0, 0, 0, 0, 0]))
    else:
        np.save(save_path, results)

    print(f"{min_rate} <= rates <= {max_rate}")
    print(f"total {results.shape[0]} std pairs gathered")


def multi_g_value_select(save_path, dest):
    proc_list = []
    proc_num = 0
    value = np.linspace(0, 0.5, 250)
    shared_dic = mp.Manager().dict()
    for i in range(0, 250, 50):
        for j in range(0, 250, 50):
            p = mp.Process(target=block_g_eval, args=(value[i:i+50], value[j:j+50], 10000, proc_num, shared_dic, dest))
            proc_list.append(p)
            proc_num += 1
            p.start()
    for proc in proc_list:
        proc.join()

    results = None
    for i in range(proc_num):
        if results is None:
            results = shared_dic[i]
        else:
            results = np.vstack((results, shared_dic[i]))

    results = results[results[:, -1] != -1]

    np.save(save_path, results)


def multi_tau_select():
    l = []
    j = 0
    names = []
    for i in range(0, 30, 5):
        save_name = "data/IVL_search/g_std_tau_for_rates" + str(j) + ".npy"
        j += 1
        p = mp.Process(target=tau_value_select, args=(i, i+5, 10, 10000, "data/IVL_search/g_std_for_rates.npy",
                                                      save_name))
        l.append(p)
        names.append(save_name)
    for proc in l:
        proc.start()
    for proc in l:
        proc.join()

    results = None
    for name in names:
        if results is None:
            results = np.load(name)
        else:
            results = np.vstack((results, np.load(name)))

    np.save("data/IVL_search/g_std_tau_for_rates_inrange.npy", results)


def multi_std_select(load_path, save_path, dest):
    proc_list = []
    proc_num = 0
    shared_dic = mp.Manager().dict()
    g_paras = np.load(load_path)
    for i in range(g_paras.shape[0]):
        #   max_value, steps, run_time, lines, shared_dic, proc_num
        p = mp.Process(target=single_std_select, args=(0.01, 50, 10000, g_paras[i], shared_dic, proc_num, dest))
        proc_list.append(p)
        proc_num += 1
        p.start()

    for proc in proc_list:
        proc.join()

    results = None
    for i in range(proc_num):
        if results is None:
            results = shared_dic[i]
        else:
            results = np.vstack((results, shared_dic[i]))

    results = results[results[:, -1] != -1]

    np.save(save_path, results)


def visualize_some(load_path):
    runtime = 1000
    h.tstop = runtime
    rates = np.load(load_path)
    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1
    vec = r1.set_up_full_recording()[0]
    for i in range(5):
        line = rates[i]
        OU_in.g_e0 = line[0]
        OU_in.g_i0 = line[1]
        OU_in.std_e = line[2]
        OU_in.std_i = line[3]
        OU_in.tau_e = line[4]
        OU_in.tau_i = line[5]


def seed_test():
    rates = np.load("data/IVL_search/g_std_tau_for_rates_binned.npy")
    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1
    vec = r1.set_up_full_recording()[0]
    h.tstop = 1000
    OU_in.new_seed(2021)
    for i in range(5):
        line = rates[0]
        OU_in.g_e0 = line[0]
        OU_in.g_i0 = line[1]
        OU_in.std_e = line[2]
        OU_in.std_i = line[3]
        OU_in.tau_e = line[4]
        OU_in.tau_i = line[5]
        # OU_in.new_seed(2021)
        h.run()
        plt.plot(np.array(vec))
        plt.savefig("testing/seed_run" + str(i) + ".png")
        plt.close()


def plot_gs(loadpath, savepath):
    # gs = np.load("data/IVL_search/g_for_rates.npy")
    gs = np.load(loadpath)

    fig = plt.figure()
    ax = fig.add_subplot(1, 15, (1, 14))
    ax1 = fig.add_subplot(1, 15, 15)
    colors = plt.cm.viridis(np.array(range(gs.shape[0])) / gs.shape[0])
    ax.scatter(gs[:, 0], gs[:, 1], c=colors)
    ax.set_xlabel(f'$g_{{e0}}$')
    ax.set_ylabel(f'$g_{{i0}}$')

    cmap = plt.cm.viridis
    norm = matplotlib.colors.Normalize(vmin=0, vmax=gs.shape[0])
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='vertical')
    cb1.set_label('Pair number')

    # plt.savefig("1_comp_plots/IVL/paraplots/gs.png")
    plt.savefig(savepath)
    plt.close()


def plot_stds(gs_path, g_std_binned_path, savepath):
    # gs = np.load("data/IVL_search/g_for_rates.npy")
    # g_std_binned = np.load("data/IVL_search/g_std_binned.npy")
    gs = np.load(gs_path)
    g_std_binned = np.load(g_std_binned_path)
    fig = plt.figure()

    ax = fig.add_subplot(1, 15, (1, 14))
    ax1 = fig.add_subplot(1, 15, 15)
    colors = plt.cm.viridis(np.array(range(gs.shape[0])) / gs.shape[0])

    for i in range(gs.shape[0]):
        curr_g_bin = g_std_binned[g_std_binned[:, 0] == i]
        for j in range(curr_g_bin.shape[0]):
            line = curr_g_bin[j]                            # [bin_num, ge, gi, stde, stdi]
            # ax.scatter(line[3], line[4], s=1, c=[colors[i]])
            ax.scatter(line[3], line[4], c=[colors[i]])

    cmap = plt.cm.viridis
    norm = matplotlib.colors.Normalize(vmin=0, vmax=gs.shape[0])
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='vertical')
    cb1.set_label('Conductance Pair number')

    ax.set_xlabel(f'$std_e$')
    ax.set_ylabel(f'$std_i$')
    # plt.savefig("1_comp_plots/IVL/paraplots/stds.png")
    plt.savefig(savepath)
    plt.close()


def plot_stds_no_overlap(division, gs_path, g_std_binned_path, base_savepath):
    # gs = np.load("data/IVL_search/g_for_rates.npy")
    # g_std_binned = np.load("data/IVL_search/g_std_binned.npy")
    gs = np.load(gs_path)
    g_std_binned = np.load(g_std_binned_path)

    for k in range(0, gs.shape[0], division):
        fig = plt.figure()
        ax = fig.add_subplot(1, 15, (1, 14))
        ax1 = fig.add_subplot(1, 15, 15)
        colors = plt.cm.viridis(np.array(range(gs.shape[0])) / gs.shape[0])
        for i in range(k, k+division, 1):
            curr_g_bin = g_std_binned[g_std_binned[:, 0] == i]
            for j in range(curr_g_bin.shape[0]):
                line = curr_g_bin[j]                            # [bin_num, ge, gi, stde, stdi]
                # ax.scatter(line[3], line[4], s=1, c=[colors[i]])
                ax.scatter(line[3], line[4], c=[colors[i]])


        cmap = plt.cm.viridis
        norm = matplotlib.colors.Normalize(vmin=0, vmax=gs.shape[0])
        cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='vertical')
        cb1.set_label('Conductance Pair number')

        ax.set_xlabel(f'$std_e$')
        ax.set_ylabel(f'$std_i$')
        # plt.savefig("1_comp_plots/IVL/paraplots/stds" + str(k) + ".png")
        plt.savefig(base_savepath + str(k) + ".png")
        plt.close()


def g_binner(g_path, savepath):
    gs = np.load(g_path)
    g_bin = None
    for i in range(gs.shape[0]):
        ge, gi = gs[i][0], gs[i][1]
        if g_bin is None:
            g_bin = np.array([[ge, gi]])
        else:
            temp = g_bin[g_bin[:, 0] == ge]
            temp = temp[temp[:, 1] == gi]
            if temp.shape[0] == 0:
                g_bin = np.vstack((g_bin, np.array([ge, gi])))

    np.save(savepath, g_bin)


def g_binned_sort(gb_path, savepath):
    gs = np.load(gb_path)
    ge = gs[:, 0]
    ge = np.sort(ge, axis=0)
    result = None
    for i in range(ge.shape[0]):
        gi = gs[gs[:, 0] == ge[i]]
        gi = np.sort(gi, axis=0)
        if result is None:
            result = gi
        else:
            temp = result[result[:, 0] == ge[i]]
            if temp.shape[0] == 0:
                result = np.vstack((result, gi))
    np.save(savepath, result)


def stds_binner(g_path, g_std_path, savepath):
    gs = np.load(g_path)
    g_std = np.load(g_std_path)
    g_bin = None
    for i in range(gs.shape[0]):
        ge, gi = gs[i][0], gs[i][1]
        for j in range(g_std.shape[0]):
            if g_std[j][0] == ge and g_std[j][1] == gi:
                row = list(g_std[j])
                row.insert(0, i)
                row = np.array(row)
                if g_bin is None:
                    g_bin = row
                else:
                    g_bin = np.vstack((g_bin, row))

    np.save(savepath, g_bin)


if __name__ == "__main__":
    mp.freeze_support()
    # testrun()
    # print(metric_testrun())
    para_explorations()

    # cond_labels = [f'$g_{{i0}}$', f'$g_{{e0}}$']
    # para_explore_conductance(0.1, 50, 10000, "data/IVL_search/score_g_max_0.1")
    # score_plotter(0.1, 50, cond_labels, "data/IVL_search/score_g_max_0.1.npy", "1_comp_plots/IVL/score_g_max_0.1.png")

    # noise_labels = [f'$std_i$', f'$std_e$']
    # para_explore_noise(0.01, 10, 10000, "data/IVL_search/score_noise_mx0.01_st10")
    # score_plotter(0.1, 10, noise_labels, "data/IVL_search/score_noise_mx0.01_st10.npy",
    #               "1_comp_plots/IVL/score_noise_mx0.01_st10.png")
    #
    # tau_labels = [f'$tau_i$', f'$tau_e$']
    # para_explore_tau(50, 50, 10000, "data/IVL_search/score_tau_max_50")
    # score_plotter(50, 50, tau_labels, "data/IVL_search/score_tau_max_50.npy", "1_comp_plots/IVL/score_tau_max_50.png")

    # multi_g_value_select("g_for_rates_dest2.npy", dest=True)
    # multi_g_value_select("g_for_rates_agm2.npy", dest=False)
    # multi_std_select(num_runs=1, load_path="g_for_rates.npy", save_path="g_std_for_rates.npy", inter_dir="")
    # stds_binner(g_path="g_for_rates.npy", g_std_path="g_std_for_rates.npy", savepath="g_std_for_rates_binned")

    # plot_gs("data/IVL_search/g_for_rates_dest.npy", "1_comp_plots/IVL/paraplots/gs_dest.png")
    # plot_gs("data/IVL_search/g_for_rates_agm.npy", "1_comp_plots/IVL/paraplots/gs_agm.png")

    # plot_gs()
    # stds_binner()
    # plot_stds()
    # plot_stds_no_overlap()

    # a = np.load("data/IVL_search/g_for_rates_agm.npy")
    # a0, a1, a2 = a[:, 0], a[:, 1], a[:, 2]
    # b = np.vstack((a1, a0, a2))
    # b = b.T
    # b00 = b[0][0]
    # b01 = b[0][1]
    # b[0][0], b[0][1] = b01, b00
    # np.save("data/IVL_search/g_for_rates_agm.npy", b)
    # plot_gs("data/IVL_search/g_for_rates_agm.npy", "1_comp_plots/IVL/paraplots/gs_agm.png")
    # plot_gs("data/IVL_search/g_for_rates_agm2.npy", "1_comp_plots/IVL/paraplots/gs_agm2.png")
    # plot_gs("data/IVL_search/g_for_rates_dest2.npy", "1_comp_plots/IVL/paraplots/gs_dest2.png")

    # a = np.load("data/IVL_search/full_search_2_agm.npy")
    # a = a[a[:, -1] != -1]
    # np.save("data/IVL_search/full_search_2_agm_fix.npy", a)
    # a = np.load("data/IVL_search/full_search_2_dest.npy")
    # a = a[a[:, -1] != -1]
    # np.save("data/IVL_search/full_search_2_dest_fix.npy", a)

    # a = np.load("data/IVL_search/g_for_rates_agm_std0.05.npy")
    # a = a[a[:, -1] != -1]
    # np.save("data/IVL_search/g_for_rates_agm_std0.05.npy", a)

    # g_binner("data/IVL_search/full_search_2_agm_fix.npy", "data/IVL_search/fs2_agm_gb.npy")
    # g_binner("data/IVL_search/full_search_2_dest_fix.npy", "data/IVL_search/fs2_dest_gb.npy")
    # g_binner("data/IVL_search/g_for_rates_agm_std0.05.npy", "data/IVL_search/g_for_rates_agm_std0.05gb.npy")

    # g_binned_sort("data/IVL_search/fs2_agm_gb.npy", "data/IVL_search/fs2_agm_gb_sorted.npy")
    # g_binned_sort("data/IVL_search/fs2_dest_gb.npy", "data/IVL_search/fs2_dest_gb_sorted.npy")
    # g_binned_sort("data/IVL_search/g_for_rates_agm_std0.05gb.npy", "data/IVL_search/g_for_rates_agm_std0.05gbs.npy")

    # plot_gs("data/IVL_search/fs2_agm_gb_sorted.npy", "1_comp_plots/IVL/paraplots/full_search_2_agm.png")
    # plot_gs("data/IVL_search/fs2_dest_gb_sorted.npy", "1_comp_plots/IVL/paraplots/full_search_2_dest.png")
    # plot_gs("data/IVL_search/g_for_rates_agm_std0.05gbs.npy", "1_comp_plots/IVL/paraplots/g_for_rates_agm_std0.05gbs.png")

    # stds_binner("data/IVL_search/fs2_agm_gb_sorted.npy", "data/IVL_search/full_search_2_agm_fix.npy",
    #             "data/IVL_search/fs2_agm_gstdb.npy")
    # stds_binner(g_path="data/IVL_search/fs2_dest_gb_sorted.npy", g_std_path="data/IVL_search/full_search_2_dest_fix.npy",
    #             savepath="data/IVL_search/fs2_dest_gstdb.npy")
    #
    # plot_stds(gs_path="data/IVL_search/fs2_agm_gb_sorted.npy", g_std_binned_path="data/IVL_search/fs2_agm_gstdb.npy",
    #           savepath="1_comp_plots/IVL/paraplots/stds_agm.png")
    # plot_stds(gs_path="data/IVL_search/fs2_dest_gb_sorted.npy", g_std_binned_path="data/IVL_search/fs2_dest_gstdb.npy",
    #           savepath="1_comp_plots/IVL/paraplots/stds_dest.png")

    # plot_stds_no_overlap(division=4, gs_path="data/IVL_search/fs2_agm_gb_sorted.npy",
    #                      g_std_binned_path="data/IVL_search/fs2_agm_gstdb.npy",
    #                      base_savepath="1_comp_plots/IVL/paraplots/stds_agm.png")
    # plot_stds_no_overlap(division=4, gs_path="data/IVL_search/fs2_dest_gb_sorted.npy",
    #                      g_std_binned_path="data/IVL_search/fs2_dest_gstdb.npy",
    #                      base_savepath="1_comp_plots/IVL/paraplots/stds_dest.png")
