import numpy as np
from neuron import h, gui
import record_1comp as r1
import efel
from IVL_helper import *
import scipy.signal as signal
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import multiprocessing as mp
from currents_visualization import *
import IVL_select
import spike_resonance_ivl_final_save as spres

### Instantiate Model ###
h.load_file("init_1comp.hoc")
h.cvode_active(0)
h.dt = 0.1
h.steps_per_ms = 10
DNQX_Hold = 0.004
TTX_Hold = -0.0290514
# cell = h.cell


def para_explorations():
    h.tstop = 1000
    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1
    line = [0.01, 0.04, 0.001, 0.009]
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
    h.run()
    v_vec = np.array(vecs[0])
    t_vec = np.array(range(len(v_vec))) * h.dt
    print(evaluator3(t_vec, v_vec))
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


def plot_gs(loadpath, savepath, plot_type="pair_num"):
    # gs = np.load("data/IVL_search/g_for_rates.npy")
    gs = np.load(loadpath)

    fig = plt.figure()
    ax = fig.add_subplot(1, 15, (1, 14))
    ax1 = fig.add_subplot(1, 15, 15)
    if plot_type == "pair_num":
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


def plot_stds(gs_path, g_std_binned_path, savepath, plot_type="pair_num"):
    # gs = np.load("data/IVL_search/g_for_rates.npy")
    # g_std_binned = np.load("data/IVL_search/g_std_binned.npy")
    gs = np.load(gs_path)
    g_std_binned = np.load(g_std_binned_path)
    fig = plt.figure()

    ax = fig.add_subplot(1, 15, (1, 14))
    ax1 = fig.add_subplot(1, 15, 15)
    colors = plt.cm.viridis(np.array(range(gs.shape[0])) / gs.shape[0])
    if plot_type == "pair_num":
        norm = matplotlib.colors.Normalize(vmin=0, vmax=gs.shape[0])
        ax.scatter(g_std_binned[:, 3], g_std_binned[:, 4], c=g_std_binned[:, 0], norm=norm)
        cmap = plt.cm.viridis
        cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='vertical')
        cb1.set_label('Conductance Pair number')
    elif plot_type == "v_mean":
        cmap = plt.cm.PiYG
        norm = matplotlib.colors.CenteredNorm(vcenter=-68.85)
        ax.scatter(g_std_binned[:, 3], g_std_binned[:, 4], c=g_std_binned[:, 5], cmap=cmap, norm=norm)

        cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='vertical')
        cb1.set_label('Sub-threshold mean voltage (mV)')

    elif plot_type == "v_var":
        cmap = plt.cm.seismic
        norm = matplotlib.colors.CenteredNorm(vcenter=9)
        ax.scatter(g_std_binned[:, 3], g_std_binned[:, 4], c=g_std_binned[:, 6], cmap=cmap, norm=norm)

        cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='vertical')
        cb1.set_label('Sub-threshold voltage variance (mV)')

    elif plot_type == "spiking_rate":
        cmap = plt.cm.OrRd
        min_rate = np.min(g_std_binned[:, 7])
        max_rate = np.max(g_std_binned[:, 7])
        ax.scatter(g_std_binned[:, 3], g_std_binned[:, 4], c=g_std_binned[:, 7], cmap=cmap, vmin=min_rate, vmax=max_rate)
        norm = matplotlib.colors.Normalize(vmin=min_rate, vmax=max_rate)
        cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='vertical')
        cb1.set_label('mean firing rate')
    ax.set_xlabel(f'$std_e$')
    ax.set_ylabel(f'$std_i$')
    # plt.savefig("1_comp_plots/IVL/paraplots/stds.png")
    plt.savefig(savepath)
    plt.close()


def plot_stds_no_overlap(division, gs_path, g_std_binned_path, base_savepath, plot_type="pair_num"):
    # gs = np.load("data/IVL_search/g_for_rates.npy")
    # g_std_binned = np.load("data/IVL_search/g_std_binned.npy")
    gs = np.load(gs_path)
    g_std_binned = np.load(g_std_binned_path)

    for k in range(0, gs.shape[0], division):
        fig = plt.figure()
        ax = fig.add_subplot(1, 15, (1, 14))
        ax1 = fig.add_subplot(1, 15, 15)

        if plot_type == "pair_num":
            cmap = plt.cm.viridis
            norm = matplotlib.colors.Normalize(vmin=0, vmax=gs.shape[0])
            curr_g_bin = g_std_binned[g_std_binned[:, 0] == k]
            for i in range(k+1, k + division, 1):
                curr_g_bin = np.vstack((curr_g_bin, g_std_binned[g_std_binned[:, 0] == i]))
            ax.scatter(curr_g_bin[:, 3], curr_g_bin[:, 4], c=curr_g_bin[:, 0], norm=norm, cmap=cmap)
            cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='vertical')
            cb1.set_label('Conductance Pair number')

        elif plot_type == "v_mean":
            cmap = plt.cm.PiYG
            norm = matplotlib.colors.CenteredNorm(vcenter=-68.85)
            curr_g_bin = g_std_binned[g_std_binned[:, 0] == k]
            for i in range(k+1, k + division, 1):
                curr_g_bin = np.vstack((curr_g_bin, g_std_binned[g_std_binned[:, 0] == i]))
            ax.scatter(curr_g_bin[:, 3], curr_g_bin[:, 4], c=curr_g_bin[:, 5], cmap=cmap, norm=norm)
            cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='vertical')
            cb1.set_label('Sub-threshold mean voltage (mV)')

        elif plot_type == "v_var":
            cmap = plt.cm.seismic
            norm = matplotlib.colors.CenteredNorm(vcenter=9)
            curr_g_bin = g_std_binned[g_std_binned[:, 0] == k]
            for i in range(k+1, k + division, 1):
                curr_g_bin = np.vstack((curr_g_bin, g_std_binned[g_std_binned[:, 0] == i]))
            ax.scatter(curr_g_bin[:, 3], curr_g_bin[:, 4], c=curr_g_bin[:, 6], cmap=cmap, norm=norm)
            cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='vertical')
            cb1.set_label('Sub-threshold voltage variance (mV)')

        elif plot_type == "spiking_rate":
            cmap = plt.cm.OrRd
            curr_g_bin = g_std_binned[g_std_binned[:, 0] == k]
            for i in range(k+1, k + division, 1):
                curr_g_bin = np.vstack((curr_g_bin, g_std_binned[g_std_binned[:, 0] == i]))
            min_rate = np.min(curr_g_bin[:, 7])
            max_rate = np.max(curr_g_bin[:, 7])
            norm = matplotlib.colors.Normalize(vmin=min_rate, vmax=max_rate)
            ax.scatter(curr_g_bin[:, 3], curr_g_bin[:, 4], c=curr_g_bin[:, 7], cmap=cmap, norm=norm)
            cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='vertical')
            cb1.set_label('mean firing rate')
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


def multi_rate_from_binned(runtime, num_div, gstb_path, savepath, save=True, seed_size=50):
    paras = np.load(gstb_path)  # [bin_num, ge, gi, stde, stdi, ...]
    slice_size = int(paras.shape[0] // num_div)
    proc_list = []
    shared_dic = mp.Manager().dict()
    for i in range(0, paras.shape[0], slice_size):
        p = mp.Process(target=single_rate_from_binned, args=(paras[i: i + slice_size], runtime, shared_dic, i, seed_size))
        p.start()
        proc_list.append(p)
    for p in proc_list:
        p.join()

    results = None
    for key in list(shared_dic):
        if shared_dic[key] is None:
            sing_result = np.array([-1, -1, -1, -1, -1, -1, -1])
        else:
            sing_result = shared_dic[key]

        if results is None:
            results = sing_result
        else:
            results = np.vstack((results, sing_result))

    if results is None:
        print("Mission failed, we will get'em next time.")
    else:
        results = results[results[:, -1] != -1]
        if save:
            np.save(savepath, results)
        else:
            return results


def single_rate_from_binned(paras, runtime, shared_dic, proc_num, seed_size=50):
    h.tstop = runtime
    seeds = np.array(range(seed_size)) + 2021
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
        OU_in.g_e0 = paras[i][1]    # [bin_num, ge, gi, stde, stdi, v_mean, v_var]
        OU_in.g_i0 = paras[i][2]
        OU_in.std_e = paras[i][3]
        OU_in.std_i = paras[i][4]
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
            results = np.zeros(8)
            results[7] = np.mean(interim)
            results[:7] = paras[i]
        else:
            line = np.zeros(8)
            line[7] = np.mean(interim)
            line[:7] = paras[i]
            results = np.vstack((results, line))
    if not (results is None):
        shared_dic[proc_num] = results


def show_examples(paras, savedir):
    # ivl_paras = np.load("data/IVL_search/g_std_tau_for_rates_binned.npy")[0]
    h.tstop = 1000
    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1
    OU_in.E_i = -87.1
    OU_in.tau_e = 12.7
    OU_in.tau_i = 12.05
    v_vec = r1.set_up_full_recording()[0]
    for i in range(len(paras)):
        OU_in.g_e0 = paras[i][2]
        OU_in.g_i0 = paras[i][3]
        OU_in.std_e = paras[i][4]
        OU_in.std_i = paras[i][5]
        h.run()
        v = np.array(v_vec)
        t = np.array(range(len(v))) * h.dt
        plt.plot(t, v)
        plt.xlabel("time (ms)")
        plt.ylabel("voltage (mV)")
        plt.savefig(f"{savedir}/example_{i}.png")
        plt.close()


def full_pre_process(dataname, dirpath, div_num=4, seed_size=50):
    pre_fix = np.load(f"{dirpath}{dataname}.npy")
    fix = pre_fix[pre_fix[:, -1] != -1]
    np.save(f"{dirpath}{dataname}_fix.npy", fix)

    g_binner(f"{dirpath}{dataname}_fix.npy", f"{dirpath}{dataname}_gb.npy")

    g_binned_sort(f"{dirpath}{dataname}_gb.npy", f"{dirpath}{dataname}_gb_sorted.npy")

    stds_binner(f"{dirpath}{dataname}_gb_sorted.npy", f"{dirpath}{dataname}_fix.npy",
                f"{dirpath}{dataname}_gstdb.npy")
    multi_rate_from_binned(10000, div_num, gstb_path=f"{dirpath}{dataname}_gstdb.npy",
                           savepath=f"{dirpath}{dataname}_gstdbr.npy", seed_size=seed_size)


def full_analysis_process(dataname, dirpath, division, save_dirpath):
    # {dirpath}{dataname}_gstdb.npy -> bin, ge, gi, stde, stdi, [v_mean, v_var]
    plot_types = ["pair_num", "v_mean", "v_var", "spiking_rate"]
    inserts = ["", "prp_wt_vmean/", "prp_wt_vvar/", "prp_wt_sr/"]
    plot_gs(f"{dirpath}{dataname}_gb_sorted.npy", f"{save_dirpath}{dataname}.png")
    for i in range(4):
        plot_stds(gs_path=f"{dirpath}{dataname}_gb_sorted.npy", g_std_binned_path=f"{dirpath}{dataname}_gstdbr.npy",
                  savepath=f"{save_dirpath}{inserts[i]}stds_{dataname}.png", plot_type=plot_types[i])

        plot_stds_no_overlap(division=division, gs_path=f"{dirpath}{dataname}_gb_sorted.npy",
                             g_std_binned_path=f"{dirpath}{dataname}_gstdbr.npy",
                             base_savepath=f"{save_dirpath}{inserts[i]}{dataname}_stds.png", plot_type=plot_types[i])


if __name__ == "__main__":
    mp.freeze_support()

    # full_process(dataname="ofs1", dirpath="data/IVL_search/", division=20,
    #              save_dirpath="1_comp_plots/IVL/paraplots/")

    # dataname = "ofs1"
    # dirpath = "data/IVL_search/"

    # save_dirpath="1_comp_plots/IVL/paraplots/ofs1/"
    # plot_stds_no_overlap(division=40, gs_path=f"{dirpath}{dataname}_gb_sorted.npy",
    #                      g_std_binned_path=f"{dirpath}{dataname}_gstdb.npy",
    #                      base_savepath=f"{save_dirpath}{dataname}_stds.png")
    #
    # save_dirpath="1_comp_plots/IVL/paraplots/ofs1/prp_wt_vmean/"
    # plot_stds_no_overlap(division=40, gs_path=f"{dirpath}{dataname}_gb_sorted.npy",
    #                      g_std_binned_path=f"{dirpath}{dataname}_gstdb.npy",
    #                      base_savepath=f"{save_dirpath}{dataname}_stds.png", plot_type="v_mean")
    # plot_stds(gs_path=f"{dirpath}{dataname}_gb_sorted.npy", g_std_binned_path=f"{dirpath}{dataname}_gstdb.npy",
    #           savepath=f"{save_dirpath}stds_{dataname}.png", plot_type="v_mean")
    #
    # save_dirpath="1_comp_plots/IVL/paraplots/ofs1/prp_wt_vvar/"
    # plot_stds_no_overlap(division=40, gs_path=f"{dirpath}{dataname}_gb_sorted.npy",
    #                      g_std_binned_path=f"{dirpath}{dataname}_gstdb.npy",
    #                      base_savepath=f"{save_dirpath}{dataname}_stds.png", plot_type="v_var")
    # plot_stds(gs_path=f"{dirpath}{dataname}_gb_sorted.npy", g_std_binned_path=f"{dirpath}{dataname}_gstdb.npy",
    #           savepath=f"{save_dirpath}stds_{dataname}.png", plot_type="v_var")
    #
    # # multi_rate_from_binned(10000, 4, gstb_path=f"{dirpath}{dataname}_gstdb.npy",
    # #                        savepath=f"{dirpath}{dataname}_gstdbr.npy")
    #
    # save_dirpath="1_comp_plots/IVL/paraplots/ofs1/prp_wt_sr/"
    # plot_stds_no_overlap(division=40, gs_path=f"{dirpath}{dataname}_gb_sorted.npy",
    #                      g_std_binned_path=f"{dirpath}{dataname}_gstdbr.npy",
    #                      base_savepath=f"{save_dirpath}{dataname}_stds.png", plot_type="spiking_rate")
    # plot_stds(gs_path=f"{dirpath}{dataname}_gb_sorted.npy", g_std_binned_path=f"{dirpath}{dataname}_gstdbr.npy",
    #           savepath=f"{save_dirpath}stds_{dataname}.png", plot_type="spiking_rate")


    # dataname = "ofs2"
    # dirpath = "data/IVL_search/"
    #
    # save_dirpath="1_comp_plots/IVL/paraplots/ofs2/"
    # plot_stds(gs_path=f"{dirpath}{dataname}_gb_sorted.npy", g_std_binned_path=f"{dirpath}{dataname}_gstdb.npy",
    #           savepath=f"{save_dirpath}stds_{dataname}.png")
    # plot_stds_no_overlap(division=40, gs_path=f"{dirpath}{dataname}_gb_sorted.npy",
    #                      g_std_binned_path=f"{dirpath}{dataname}_gstdb.npy",
    #                      base_savepath=f"{save_dirpath}{dataname}_stds.png")

    # save_dirpath="1_comp_plots/IVL/paraplots/ofs2/prp_wt_vmean/"
    # plot_stds(gs_path=f"{dirpath}{dataname}_gb_sorted.npy", g_std_binned_path=f"{dirpath}{dataname}_gstdb.npy",
    #           savepath=f"{save_dirpath}stds_{dataname}.png", plot_type="v_mean")
    # plot_stds_no_overlap(division=20, gs_path=f"{dirpath}{dataname}_gb_sorted.npy",
    #                      g_std_binned_path=f"{dirpath}{dataname}_gstdb.npy",
    #                      base_savepath=f"{save_dirpath}{dataname}_stds.png", plot_type="v_mean")

    # save_dirpath="1_comp_plots/IVL/paraplots/ofs2/prp_wt_vvar/"
    # plot_stds(gs_path=f"{dirpath}{dataname}_gb_sorted.npy", g_std_binned_path=f"{dirpath}{dataname}_gstdb.npy",
    #           savepath=f"{save_dirpath}stds_{dataname}.png", plot_type="v_var")
    # plot_stds_no_overlap(division=20, gs_path=f"{dirpath}{dataname}_gb_sorted.npy",
    #                      g_std_binned_path=f"{dirpath}{dataname}_gstdb.npy",
    #                      base_savepath=f"{save_dirpath}{dataname}_stds.png", plot_type="v_var")

    # ofs2_100b_gb = np.load("data/IVL_search/ofs2_gb_sorted.npy")[:100]
    # np.save("data/IVL_search/ofs2_100b_gb_sorted.npy", ofs2_100b_gb)
    # ofs2_gstdb = np.load("data/IVL_search/ofs2_gstdb.npy")
    # ofs2_100b_gstdb = ofs2_gstdb[ofs2_gstdb[:, 0] < 100]
    # np.save("data/IVL_search/ofs2_100b_gstdb.npy", ofs2_100b_gstdb)

    # dataname = "ofs2_100b"
    # dirpath = "data/IVL_search/"

    # save_dirpath = "1_comp_plots/IVL/paraplots/ofs2_100b/"
    # plot_gs(f"{dirpath}{dataname}_gb_sorted.npy", f"{save_dirpath}{dataname}.png")
    # plot_stds(gs_path=f"{dirpath}{dataname}_gb_sorted.npy", g_std_binned_path=f"{dirpath}{dataname}_gstdb.npy",
    #           savepath=f"{save_dirpath}stds_{dataname}.png")
    # plot_stds_no_overlap(division=10, gs_path=f"{dirpath}{dataname}_gb_sorted.npy",
    #                      g_std_binned_path=f"{dirpath}{dataname}_gstdb.npy",
    #                      base_savepath=f"{save_dirpath}{dataname}_stds.png")

    # save_dirpath="1_comp_plots/IVL/paraplots/ofs2_100b/prp_wt_vmean/"
    # plot_stds(gs_path=f"{dirpath}{dataname}_gb_sorted.npy", g_std_binned_path=f"{dirpath}{dataname}_gstdb.npy",
    #           savepath=f"{save_dirpath}stds_{dataname}.png", plot_type="v_mean")
    # plot_stds_no_overlap(division=10, gs_path=f"{dirpath}{dataname}_gb_sorted.npy",
    #                      g_std_binned_path=f"{dirpath}{dataname}_gstdb.npy",
    #                      base_savepath=f"{save_dirpath}{dataname}_stds.png", plot_type="v_mean")

    # save_dirpath="1_comp_plots/IVL/paraplots/ofs2_100b/prp_wt_vvar/"
    # plot_stds(gs_path=f"{dirpath}{dataname}_gb_sorted.npy", g_std_binned_path=f"{dirpath}{dataname}_gstdb.npy",
    #           savepath=f"{save_dirpath}stds_{dataname}.png", plot_type="v_var")
    # plot_stds_no_overlap(division=10, gs_path=f"{dirpath}{dataname}_gb_sorted.npy",
    #                      g_std_binned_path=f"{dirpath}{dataname}_gstdb.npy",
    #                      base_savepath=f"{save_dirpath}{dataname}_stds.png", plot_type="v_var")

    # save_dirpath="1_comp_plots/IVL/paraplots/ofs2_100b/prp_wt_sr/"
    # plot_stds(gs_path=f"{dirpath}{dataname}_gb_sorted.npy", g_std_binned_path=f"{dirpath}{dataname}_gstdbr.npy",
    #           savepath=f"{save_dirpath}stds_{dataname}.png", plot_type="spiking_rate")
    # plot_stds_no_overlap(division=10, gs_path=f"{dirpath}{dataname}_gb_sorted.npy",
    #                      g_std_binned_path=f"{dirpath}{dataname}_gstdbr.npy",
    #                      base_savepath=f"{save_dirpath}{dataname}_stds.png", plot_type="spiking_rate")

    # full_pre_process("ps3", "data/IVL_search/")
    # full_analysis_process("ps3", "data/IVL_search/", 4, "1_comp_plots/IVL/paraplots/ps3/")
    # multi_rate_from_binned(10000, 4, gstb_path=f"data/IVL_search/ps3_gstdb.npy",
    #                        savepath=f"data/IVL_search/ps3_gstdbr.npy")
    # full_pre_process("ps4", "data/IVL_search/")
    # full_analysis_process("ps4", "data/IVL_search/", 4, "1_comp_plots/IVL/paraplots/ps4/")
    # full_pre_process("ps5", "", div_num=128)
    # full_pre_process("ofs3", "", div_num=128, seed_size=1)

    lst = [10, 40, 70, 120, 170, 220, 260, 300, 330, 350]
    img_dir = "1_comp_plots/IVL/spike_res/spres_ps5"
    para_path = "data/IVL_select/ps5_sl3_stats.npy"
    para_sorted = spres.paras_sorter(para_path)
    show_examples(para_sorted[lst], img_dir)
