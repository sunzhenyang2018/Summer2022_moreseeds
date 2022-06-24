from STA import *


def fr_(norm, fr_rate, num_states, load_dir, save_dir, numpts=None, raw=True, directed=False):
    fig = plt.figure()
    labels = ['IKa', 'IKdrf', 'Im', 'Il', 'INa', 'Ih']
    for i in range(6):
        ax = fig.add_subplot(3, 2, i + 1)
        collected = []
        for j in range(num_states):
            if numpts is not None:
                avg_std = np.load(f"{load_dir}/{j}/sta_at_fr/st{j}_fr_{fr_rate}_{numpts}s_inh.npy")
            else:
                avg_std = np.load(f"{load_dir}/{j}/sta_at_fr/st{j}_fr_{fr_rate}_inh.npy")
            if norm:
                abs_total = np.sum(abs(avg_std[0][1:]), axis=0)[199:-99]
            else:
                abs_total = 1
            normalized = avg_std[0][i + 1][199:-99] / abs_total
            collected.append(normalized)
        collected_norm = np.array(collected)
        abs_col = np.abs(collected_norm)
        if directed:
            direction = collected_norm / np.abs(collected_norm)
            if np.amin(direction) < 0:
                assert np.amax(collected_norm) <= 0 or i == 3
        else:
            direction = 1
        if np.amin(direction) < 0:
            assert np.amax(collected_norm) <= 0 or i == 3
        if not raw:
            collected_norm = (abs_col - np.amin(abs_col)) / (np.amax(abs_col) - np.amin(abs_col)) * direction
        t_vec = (np.array(range(1 - avg_std.shape[2], 1, 1)) * h.dt)[199:-99]
        for j in range(num_states):
            ax.plot(t_vec, collected_norm[j], label=labels[i])
        start, end = np.amin(collected_norm), np.amax(collected_norm)
        # ax.yaxis.set_ticks(np.arange(round(start, 2), round(end, 2)+0.001, round((end - start)/5, 2)))
        start, end = np.amin(t_vec), np.amax(t_vec)
        # ax.xaxis.set_ticks(np.arange(round(start), round(end)+0.001, round((end - start) / 5)))

    fig.suptitle(f"combined fr {fr_rate} inh {'norm' if norm else ''}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{'directed' if directed else 'undirected'}/combined_fr{fr_rate}"
                f"{'_norm_' if norm else '_'}{num_states}states{'' if raw else '_norm'}.png")

    plt.close()


def spread_fr_(norm, fr_rate, num_states, load_dir, save_dir, numpts=None, txt_file=None, raw=True, directed=False):
    fig = plt.figure()
    labels = ['IKa', 'IKdrf', 'Im', 'Il', 'INa', 'Ih']
    if txt_file is not None:
        txt_file.write(f"\n FR {fr_rate}:\n")
    for i in range(6):
        ax = fig.add_subplot(3, 2, i + 1)
        collected = []
        for j in range(num_states):
            if numpts is not None:
                avg_std = np.load(f"{load_dir}/{j}/sta_at_fr/st{j}_fr_{fr_rate}_{numpts}s_inh.npy")
            else:
                avg_std = np.load(f"{load_dir}/{j}/sta_at_fr/st{j}_fr_{fr_rate}_inh.npy")
            if norm:
                abs_total = np.sum(abs(avg_std[0][1:]), axis=0)[199:-99]
            else:
                abs_total = 1
            normalized = avg_std[0][i + 1][199:-99] / abs_total
            collected.append(normalized)
        t_vec = (np.array(range(1 - avg_std.shape[2], 1, 1)) * h.dt)[199:-99]
        collected_norm = np.array(collected)
        abs_col = np.abs(collected_norm)
        if directed:
            direction = collected_norm / np.abs(collected_norm)
            if np.amin(direction) < 0:
                assert np.amax(collected_norm) <= 0 or i == 3
        else:
            direction = 1
        if not raw:
            collected_norm = (abs_col - np.amin(abs_col)) / (np.amax(abs_col) - np.amin(abs_col)) * direction
        c_mean = np.mean(collected_norm, axis=0)
        c_std = np.sqrt(np.var(collected_norm, axis=0))
        ax.plot(t_vec, c_mean)
        ax.plot(t_vec, c_mean - c_std, linestyle='--')
        ax.plot(t_vec, c_mean + c_std, linestyle='--')
        if txt_file is not None:
            txt_file.write(f"{labels[i]} total spread: {np.sum(2*c_std)}\n")
        start, end = ax.get_ylim()
        # ax.yaxis.set_ticks(np.arange(round(start, 2), round(end, 2), round((end - start) / 5, 2)))
        start, end = np.amin(t_vec), np.amax(t_vec)
        # ax.xaxis.set_ticks(np.arange(round(start), round(end)+0.001, round((end - start)/5)))

    fig.suptitle(f"combined fr {fr_rate} inh {'norm' if norm else ''}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{'directed' if directed else 'undirected'}/combined_spd_fr{fr_rate}"
                f"{'_norm_' if norm else '_'}{num_states}states{'' if raw else '_norm'}.png")

    plt.close()


def test_coloring(norm, fr_rate, num_states, load_dir, save_dir, numpts=None, raw=True):
    fig = plt.figure()
    labels = ['IKa', 'IKdrf', 'Im', 'Il', 'INa', 'Ih']
    for i in [2]:
        ax = fig.add_subplot(111)
        collected = []
        for j in range(num_states):
            if numpts is not None:
                avg_std = np.load(f"{load_dir}/{j}/sta_at_fr/st{j}_fr_{fr_rate}_{numpts}s_inh.npy")
            else:
                avg_std = np.load(f"{load_dir}/{j}/sta_at_fr/st{j}_fr_{fr_rate}_inh.npy")
            if norm:
                abs_total = np.sum(abs(avg_std[0][1:]), axis=0)[199:-99]
            else:
                abs_total = 1
            normalized = avg_std[0][i + 1][199:-99] / abs_total
            collected.append(normalized)
        collected_norm = np.array(collected)
        if not raw:
            collected_norm = (collected - np.amin(collected)) / (np.amax(collected) - np.amin(collected))
        t_vec = (np.array(range(1 - avg_std.shape[2], 1, 1)) * h.dt)[199:-99]
        for j in range(num_states):
            ax.plot(t_vec, collected_norm[j], label=f"state {j}")
        start, end = np.amin(collected_norm), np.amax(collected_norm)
        # ax.yaxis.set_ticks(np.arange(round(start, 2), round(end, 2)+0.001, round((end - start)/5, 2)))
        start, end = np.amin(t_vec), np.amax(t_vec)
        # ax.xaxis.set_ticks(np.arange(round(start), round(end)+0.001, round((end - start) / 5)))

    fig.suptitle(f"combined fr {fr_rate} inh {'norm' if norm else ''} Test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/Test_color_combined_fr{fr_rate}{'_norm_' if norm else '_'}{num_states}states{'' if raw else '_norm'}")

    plt.close()


def norm_single(fr_rate, num_states, load_dir, save_dir, curr_lst, norm=True, numpts=None,
                directed=False, interval=(-180, -10)):
    summed = _collect_helper(fr_rate, curr_lst, num_states, numpts, load_dir, directed, interval, norm)
    t_vec = np.linspace(interval[0], interval[1], summed.shape[1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for j in range(num_states):
        ax.plot(t_vec, summed[j])
    start, end = np.amin(summed), np.amax(summed)
    # ax.yaxis.set_ticks(np.arange(round(start, 2), round(end, 2) + 0.001, round((end - start) / 10, 2)))
    start, end = np.amin(t_vec), np.amax(t_vec)
    # ax.xaxis.set_ticks(np.arange(round(start), round(end) + 0.001, round((end - start) / 10)))
    names = ""
    for i in curr_lst:
        names += labels[i]
    fig.suptitle(f"Summed Directed Normalized Contribution of {names} at {fr_rate} hz fr")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{('directed' if directed else 'undirected') if norm else 'raw'}"
                f"/{names}_fr{fr_rate}_{num_states}states.png")

    plt.close()


def spread_norm_single(fr_rate, num_states, load_dir, save_dir, curr_lst, norm=True, numpts=None, txt_file=None, directed=False,
                       interval=(-180, -10)):
    summed = _collect_helper(fr_rate, curr_lst, num_states, numpts, load_dir, directed, interval, norm)

    t_vec = np.linspace(interval[0], interval[1], summed.shape[1])
    fig = plt.figure()
    ax = fig.add_subplot(111)

    c_mean = np.mean(summed, axis=0)
    c_std = np.sqrt(np.var(summed, axis=0))
    ax.plot(t_vec, c_mean)
    ax.plot(t_vec, c_mean - c_std, linestyle='--')
    ax.plot(t_vec, c_mean + c_std, linestyle='--')

    names = ""
    for i in curr_lst:
        names += labels[i]

    if txt_file is not None:
        txt_file.write(f"\n FR {fr_rate}:\n")
        txt_file.write(f"{names} total spread: {np.sum(2 * c_std)}\n")

    start, end = np.amin(summed), np.amax(summed)
    # ax.yaxis.set_ticks(np.arange(round(start, 2), round(end, 2) + 0.001, round((end - start) / 10, 2)))
    start, end = np.amin(t_vec), np.amax(t_vec)
    # ax.xaxis.set_ticks(np.arange(round(start), round(end) + 0.001, round((end - start) / 10)))

    fig.suptitle(f"Summed Directed Normalized Contribution of {names} at {fr_rate} hz fr")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{('directed' if directed else 'undirected') if norm else 'raw'}"
                f"/{names}_spd_fr{fr_rate}_{num_states}states.png")

    plt.close()


def spread_summary(txt_filepath, fr_array, savepath, para_fit=False):
    spreads = []
    txt_file = open(txt_filepath, 'r')
    for line in txt_file:
        if 'spread' in line:
            for i in range(len(line)):
                if line[i].isdigit():
                    spreads.append(float(line[i:-1]))
                    break

    labels = [str(i) for i in fr_array]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if para_fit:
        x_dummy = np.array(range(len(spreads)))
        ax.scatter(x_dummy, spreads)
        coefs = np.polyfit(x_dummy, spreads, deg=2)
        fit_line = coefs[0] * x_dummy**2 + coefs[1] * x_dummy + coefs[2]
        ax.plot(x_dummy, fit_line, color='red')
        ax.scatter(np.argmin(fit_line), np.amin(fit_line), color='green')
    else:
        ax.plot(spreads)
    ax.set_title("Total Spread")
    ax.set_xticks(list(range(len(spreads))), labels, rotation=45)
    plt.savefig(savepath)
    plt.close()


def norm_linearfit(fr_rate, num_states, load_dir, save_dir, curr_lst, norm=True, numpts=None, txt_file=None, directed=False,
                       interval=(-180, -10)):
    summed = _collect_helper(fr_rate, curr_lst, num_states, numpts, load_dir, directed, interval, norm)

    t_vec = np.linspace(interval[0], interval[1], summed.shape[1])
    fig = plt.figure()
    ax = fig.add_subplot(111)

    c_mean = np.mean(summed, axis=0)
    coefs = np.polyfit(t_vec, c_mean, deg=1)
    ax.plot(t_vec, c_mean)
    ax.plot(t_vec, t_vec*coefs[0] + coefs[1])
    names = ""
    for i in curr_lst:
        names += labels[i]

    if txt_file is not None:
        txt_file.write(f"\n FR {fr_rate}:\n")
        txt_file.write(f"{names} slope: {coefs[0]}\n")
    start, end = np.amin(summed), np.amax(summed)
    # ax.yaxis.set_ticks(np.arange(round(start, 2), round(end, 2) + 0.001, round((end - start) / 10, 2)))
    start, end = np.amin(t_vec), np.amax(t_vec)
    # ax.xaxis.set_ticks(np.arange(round(start), round(end) + 0.001, round((end - start) / 10)))

    fig.suptitle(f"Linear fit of mean {'directed' if directed else 'undirected'} "
                 f"{'Normalized' if norm else 'raw proportion'} Contribution of {names} at {fr_rate} hz fr")
    plt.tight_layout()

    plt.savefig(f"{save_dir}/{('directed' if directed else 'undirected') if norm else 'raw'}"
                f"/{names}_fit_fr{fr_rate}_{num_states}states.png")

    plt.close()


def slope_summary(txt_filepath, fr_array, savepath, lin_fit=False):
    slopes = []
    txt_file = open(txt_filepath, 'r')
    for line in txt_file:
        if 'slope' in line:
            for i in range(len(line)):
                if line[i].isdigit() or (line[i] == "-" and line[i+1].isdigit()):
                    slopes.append(float(line[i:-1]))
                    break

    labels = [str(i) for i in fr_array]
    slopes = np.array(slopes)
    fr_array = np.array(fr_array)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if lin_fit:
        theta_start = np.argwhere(fr_array < 3).reshape((-1,))[-1] + 1
        theta_end = np.argwhere(fr_array > 12).reshape((-1,))[0] - 1

        x_dummy = np.array(range(len(slopes)))
        ax.scatter(x_dummy, slopes)
        coefs = np.polyfit(x_dummy, slopes, deg=1)
        fitline = x_dummy * coefs[0] + coefs[1]
        ax.plot(x_dummy, fitline)
        ax.vlines([theta_start, theta_end], np.amin(slopes), np.amax(slopes), linestyle='dashed')
    else:
        ax.plot(slopes)
    ax.set_title("Fr vs slope")
    ax.set_xticks(list(range(len(slopes))), labels, rotation=45)
    plt.savefig(savepath)
    plt.close()


def _collect_helper(fr_rate, curr_lst, num_states, numpts, load_dir, directed, interval, norm):
    labels = ['IKa', 'IKdrf', 'Im', 'Il', 'INa', 'Ih']
    summed = None
    slice_start = int(interval[0] / h.dt) + 1
    slice_end = int(interval[1] / h.dt) + 1
    for i in curr_lst:
        collected = []
        for j in range(num_states):
            if numpts is not None:
                avg_std = np.load(f"{load_dir}/{j}/sta_at_fr/st{j}_fr_{fr_rate}_{numpts}s_inh.npy")
            else:
                avg_std = np.load(f"{load_dir}/{j}/sta_at_fr/st{j}_fr_{fr_rate}_inh.npy")

            abs_total = np.sum(abs(avg_std[0][1:]), axis=0)[slice_start:slice_end]
            normalized = avg_std[0][i + 1][slice_start:slice_end] / abs_total
            collected.append(normalized)
        collected_norm = np.array(collected)
        abs_col = np.abs(collected_norm)
        if norm:
            if directed:
                direction = collected_norm / np.abs(collected_norm)
            else:
                direction = 1

            collected_norm = (abs_col - np.amin(abs_col)) / (np.amax(abs_col) - np.amin(abs_col)) * direction

        if summed is None:
            summed = collected_norm
        else:
            summed += collected_norm

    return summed


if __name__ == "__main__":
    labels = ['IKa', 'IKdrf', 'Im', 'Il', 'INa', 'Ih']
    # fr_(norm=True, fr_rate=9, num_states=5, save_dir="1_comp_plots/STA/ps1_optw2_sl3", load_dir="spres_ps1_optw2_sl3")
    lst = [0.5, 1.0, 2.0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21.0, 22.0, 23.0, 24.0, 25]
    # for directed in [True, False]:
    #     txt_file1 = open(f"1_comp_plots/STA/ps5/{'directed' if directed else 'undirected'}/fr_spd.txt", 'w')
    #     txt_file2 = open(f"1_comp_plots/STA/ps5/{'directed' if directed else 'undirected'}/fr_spd_norm.txt", 'w')
    #
    #     for fr_rate in lst:
    #         fr_(norm=True, fr_rate=fr_rate, num_states=10, save_dir="1_comp_plots/STA/ps5",
    #             load_dir="data/spres_ps5", numpts=50, directed=directed)
    #         spread_fr_(norm=True, fr_rate=fr_rate, num_states=10, save_dir="1_comp_plots/STA/ps5",
    #                    load_dir="data/spres_ps5", numpts=50, txt_file=txt_file1, directed=directed)
    #         fr_(norm=True, fr_rate=fr_rate, num_states=10, save_dir="1_comp_plots/STA/ps5",
    #             load_dir="data/spres_ps5", numpts=50, raw=False, directed=directed)
    #         spread_fr_(norm=True, fr_rate=fr_rate, num_states=10, save_dir="1_comp_plots/STA/ps5",
    #                    load_dir="data/spres_ps5", numpts=50, txt_file=txt_file2, raw=False, directed=directed)
    #     txt_file1.close()
    #     txt_file2.close()
    #
    # # test_coloring(norm=True, fr_rate=7, num_states=10, save_dir="1_comp_plots/STA/ps5",
    # #               load_dir="data/spres_ps5", numpts=50, raw=False)
    #
    # txt_file3 = open(f"1_comp_plots/STA/ps5_sum/directed/sum_spd.txt", 'w')
    # txt_file4 = open(f"1_comp_plots/STA/ps5_sum/undirected/sum_spd.txt", 'w')
    # for fr_rate in lst:
    #     norm_single(fr_rate, num_states=10, load_dir="data/spres_ps5", save_dir="1_comp_plots/STA/ps5_sum",
    #                 curr_lst=[2,5], numpts=50, directed=True)
    #     spread_norm_single(fr_rate, num_states=10, load_dir="data/spres_ps5", save_dir="1_comp_plots/STA/ps5_sum",
    #                        curr_lst=[2,5], numpts=50, txt_file=txt_file3, directed=True)
    #     norm_single(fr_rate, num_states=10, load_dir="data/spres_ps5", save_dir="1_comp_plots/STA/ps5_sum",
    #                 curr_lst=[2,5], numpts=50, directed=False)
    #     spread_norm_single(fr_rate, num_states=10, load_dir="data/spres_ps5", save_dir="1_comp_plots/STA/ps5_sum",
    #                        curr_lst=[2,5], numpts=50, txt_file=txt_file4, directed=False)
    # txt_file3.close()
    # txt_file4.close()
    #
    # spread_summary("1_comp_plots/STA/ps5_sum/directed/sum_spd.txt", lst,
    #                "1_comp_plots/STA/ps5_sum_directed_sum_spd.png")
    # spread_summary("1_comp_plots/STA/ps5_sum/undirected/sum_spd.txt", lst,
    #                "1_comp_plots/STA/ps5_sum_undirected_sum_spd.png")
    #
    # interval = (-180, -20)
    # for i in [2, 5]:
    #     txt_file5 = open(f"1_comp_plots/STA/ps5_sum/directed/{labels[i]}_spd.txt", 'w')
    #     txt_file6 = open(f"1_comp_plots/STA/ps5_sum/undirected/{labels[i]}_spd.txt", 'w')
    #     txt_file7 = open(f"1_comp_plots/STA/ps5_sum/raw/{labels[i]}_spd.txt", 'w')
    #     for fr_rate in lst:
    #         norm_single(fr_rate, num_states=10, load_dir="data/spres_ps5", save_dir="1_comp_plots/STA/ps5_sum",
    #                     curr_lst=[i], numpts=50, directed=True, interval=(-180, -20))
    #         spread_norm_single(fr_rate, num_states=10, load_dir="data/spres_ps5", save_dir="1_comp_plots/STA/ps5_sum",
    #                            curr_lst=[i], numpts=50, txt_file=txt_file5, directed=True, interval=(-180, -20))
    #         norm_single(fr_rate, num_states=10, load_dir="data/spres_ps5", save_dir="1_comp_plots/STA/ps5_sum",
    #                     curr_lst=[i], numpts=50, directed=False, interval=(-180, -20))
    #         spread_norm_single(fr_rate, num_states=10, load_dir="data/spres_ps5", save_dir="1_comp_plots/STA/ps5_sum",
    #                            curr_lst=[i], numpts=50, txt_file=txt_file6, directed=False, interval=(-180, -20))
    #
    #         norm_single(fr_rate, num_states=10, load_dir="data/spres_ps5", save_dir="1_comp_plots/STA/ps5_sum",
    #                     curr_lst=[i], numpts=50, directed=False, norm=False, interval=(-180, -20))
    #         spread_norm_single(fr_rate, num_states=10, load_dir="data/spres_ps5", save_dir="1_comp_plots/STA/ps5_sum",
    #                            curr_lst=[i], numpts=50, txt_file=txt_file7, directed=False, norm=False, interval=(-180, -20))
    #     txt_file5.close()
    #     txt_file6.close()
    #     txt_file7.close()
    # spread_summary("1_comp_plots/STA/ps5_sum/directed/im_spd.txt", lst,
    #                "1_comp_plots/STA/ps5_sum_directed_im_spd.png")
    # spread_summary("1_comp_plots/STA/ps5_sum/undirected/im_spd.txt", lst,
    #                "1_comp_plots/STA/ps5_sum_undirected_im_spd.png")
    #
    # spread_summary("1_comp_plots/STA/ps5_sum/directed/ih_spd.txt", lst,
    #                "1_comp_plots/STA/ps5_sum_directed_ih_spd.png")
    # spread_summary("1_comp_plots/STA/ps5_sum/undirected/ih_spd.txt", lst,
    #                "1_comp_plots/STA/ps5_sum_undirected_ih_spd.png")

    # for i in [2, 5]:
    #     txt_file7 = open(f"1_comp_plots/STA/ps5_linear_fit/directed/{labels[i]}_fit.txt", 'w')
    #     txt_file8 = open(f"1_comp_plots/STA/ps5_linear_fit/undirected/{labels[i]}_fit.txt", 'w')
    #     txt_file9 = open(f"1_comp_plots/STA/ps5_linear_fit/raw/{labels[i]}_fit.txt", 'w')
    #     for fr_rate in lst:
    #         norm_linearfit(fr_rate, num_states=10, load_dir="data/spres_ps5", save_dir="1_comp_plots/STA/ps5_linear_fit",
    #                        curr_lst=[i], numpts=50, txt_file=txt_file7, directed=True, interval=(-180, -20), norm=True)
    #         norm_linearfit(fr_rate, num_states=10, load_dir="data/spres_ps5", save_dir="1_comp_plots/STA/ps5_linear_fit",
    #                        curr_lst=[i], numpts=50, txt_file=txt_file8, directed=False, interval=(-180, -20), norm=True)
    #         norm_linearfit(fr_rate, num_states=10, load_dir="data/spres_ps5", save_dir="1_comp_plots/STA/ps5_linear_fit",
    #                        curr_lst=[i], numpts=50, txt_file=txt_file9, interval=(-180, -20), norm=False)
    #
    #     txt_file7.close()
    #     txt_file8.close()
    #     txt_file9.close()
    # slope_summary("1_comp_plots/STA/ps5_linear_fit/undirected/ih_fit.txt", lst,
    #               "1_comp_plots/STA/ps5_undirected_ih_fit.png")
    # slope_summary("1_comp_plots/STA/ps5_linear_fit/directed/ih_fit.txt", lst,
    #               "1_comp_plots/STA/ps5_directed_ih_fit.png")
    # slope_summary("1_comp_plots/STA/ps5_linear_fit/raw/ih_fit.txt", lst,
    #               "1_comp_plots/STA/ps5_raw_ih_fit.png")
    #
    # slope_summary("1_comp_plots/STA/ps5_linear_fit/undirected/im_fit.txt", lst,
    #               "1_comp_plots/STA/ps5_undirected_im_fit.png")
    # slope_summary("1_comp_plots/STA/ps5_linear_fit/directed/im_fit.txt", lst,
    #               "1_comp_plots/STA/ps5_directed_im_fit.png")
    # slope_summary("1_comp_plots/STA/ps5_linear_fit/raw/im_fit.txt", lst,
    #               "1_comp_plots/STA/ps5_raw_im_fit.png")

    # txt_file7 = open(f"1_comp_plots/STA/ps5_linear_fit/directed/hm_fit.txt", 'w')
    # txt_file8 = open(f"1_comp_plots/STA/ps5_linear_fit/undirected/hm_fit.txt", 'w')
    # txt_file9 = open(f"1_comp_plots/STA/ps5_linear_fit/raw/hm_fit.txt", 'w')
    # for fr_rate in lst:
    #     norm_linearfit(fr_rate, num_states=10, load_dir="data/spres_ps5", save_dir="1_comp_plots/STA/ps5_linear_fit",
    #                    curr_lst=[2, 5], numpts=50, txt_file=txt_file7, directed=True, interval=(-180, -20), norm=True)
    #     norm_linearfit(fr_rate, num_states=10, load_dir="data/spres_ps5", save_dir="1_comp_plots/STA/ps5_linear_fit",
    #                    curr_lst=[2, 5], numpts=50, txt_file=txt_file8, directed=False, interval=(-180, -20), norm=True)
    #     norm_linearfit(fr_rate, num_states=10, load_dir="data/spres_ps5", save_dir="1_comp_plots/STA/ps5_linear_fit",
    #                    curr_lst=[2, 5], numpts=50, txt_file=txt_file9, interval=(-180, -20), norm=False)
    #
    # txt_file7.close()
    # txt_file8.close()
    # txt_file9.close()

    # slope_summary("1_comp_plots/STA/ps5_linear_fit/undirected/hm_fit.txt", lst,
    #               "1_comp_plots/STA/ps5_undirected_hm_fit.png")
    # slope_summary("1_comp_plots/STA/ps5_linear_fit/directed/hm_fit.txt", lst,
    #               "1_comp_plots/STA/ps5_directed_hm_fit.png")
    # slope_summary("1_comp_plots/STA/ps5_linear_fit/raw/hm_fit.txt", lst,
    #               "1_comp_plots/STA/ps5_raw_hm_fit.png")

    # spread_summary("1_comp_plots/STA/ps5_sum/directed/sum_spd.txt", lst,
    #                "1_comp_plots/STA/ps5_sum_directed_sum_spd.png", para_fit=True)
    # spread_summary("1_comp_plots/STA/ps5_sum/undirected/sum_spd.txt", lst,
    #                "1_comp_plots/STA/ps5_sum_undirected_sum_spd.png", para_fit=True)

    slope_summary("1_comp_plots/STA/ps5_linear_fit/undirected/hm_fit.txt", lst,
                  "1_comp_plots/STA/ps5_undirected_hm_fit_sc.png", lin_fit=True)
    slope_summary("1_comp_plots/STA/ps5_linear_fit/directed/hm_fit.txt", lst,
                  "1_comp_plots/STA/ps5_directed_hm_fit_sc.png", lin_fit=True)
    slope_summary("1_comp_plots/STA/ps5_linear_fit/raw/hm_fit.txt", lst,
                  "1_comp_plots/STA/ps5_raw_hm_fit_sc.png", lin_fit=True)
