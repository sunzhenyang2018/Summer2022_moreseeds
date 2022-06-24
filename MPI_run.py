from neuron import h
import record_1comp as r1
from IVL_helper import *
import IVL_select
import numpy as np
import mpi4py.MPI as mpi

comm = mpi.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"this is node {rank} out of {size}\n")

# Instantiate Model
h.load_file("init_1comp.hoc")
h.cvode_active(0)
h.dt = 0.1
h.steps_per_ms = 10
DNQX_Hold = 0.004
TTX_Hold = -0.0290514


def opt_func2(i1, i2, ge_array, gi_array, stde_array, stdi_array, dic, dest):
    ge_value = ge_array
    gi_value = gi_array
    stde_value = stde_array
    stdi_value = stdi_array

    runtime = 10000
    h.tstop = runtime

    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1

    if not dest:
        OU_in.tau_e = 12.7
        OU_in.tau_i = 12.05

    v_hvec = r1.set_up_full_recording()[0]

    t_vec = np.array((range(int(runtime / h.dt) + 1))) * h.dt

    results = None
    num3 = stde_array.size
    num4 = stdi_array.size
    for j1 in range(num3):
        for j2 in range(num4):
            OU_in.g_e0 = ge_value[i1]
            OU_in.g_i0 = gi_value[i2]
            OU_in.std_e = stde_value[j1]
            OU_in.std_i = stdi_value[j2]
            h.run()
            v_vec = np.array(v_hvec)
            score = evaluator3(t_vec, v_vec)

            if score == 4:
                v_mean, v_var = IVL_select.sub_thresh_calc(t_vec, v_vec)
                ef_list = ["peak_indices"]
                efel.api.setThreshold(-30)
                efel.api.setDerivativeThreshold(1)
                trace = {'V': v_vec, 'T': t_vec, 'stim_start': [0], 'stim_end': [t_vec[-1]]}
                features = efel.getFeatureValues([trace], ef_list)[0]
                spike_rate = len(features['peak_indices']) / t_vec[-1] * 1000
                value_array = np.array([ge_value[i1], gi_value[i2], stde_value[j1],
                                        stdi_value[j2], v_mean, v_var, spike_rate])
                if results is None:
                    results = value_array
                else:
                    results = np.vstack((results, value_array))

    dic[(i1, i2)] = results


def index_distributor(num1, num2):
    total_num_run = num1 * num2     # total number of runs
    step = total_num_run // size
    leftover_run_nums = list(range(step*size, total_num_run))     # the "index" of particular run, starting from 0
    run_nums = list(range(step*rank, step*rank+step, 1))    # the indices of run at this core
    if rank < len(leftover_run_nums):
        run_nums.append(leftover_run_nums[rank])
    run_nums = np.array(run_nums)
    indices = np.zeros((run_nums.size, 2))          # indices in terms of i1 i2
    indices[:, 0] = run_nums // num2
    indices[:, 1] = run_nums % num2
    return indices


def universal_runner(i1i2, paras, dest):
    dic = dict()
    for i in range(i1i2.shape[0]):
        opt_func2(i1i2[i][0], i1i2[i][1], paras[0], paras[1], paras[2], paras[3], dic, dest=dest)

    results = None
    for key in list(dic):
        if dic[key] is None:
            sing_result = np.array([-1, -1, -1, -1, -1, -1, -1])  # ge, gi, stde, stdi
        else:
            sing_result = dic[key]

        if results is None:
            results = sing_result
        else:
            results = np.vstack((results, sing_result))

    results = results[results[:, -1] != -1]
    return results


if __name__ == "__main__":
    # always work with process more than available cores
    num = 50
    # num = 10
    paras = np.array([np.linspace(0.003, 0.006, num), np.linspace(0.008, 0.012, num),
                      np.linspace(0, 0.0016, num), np.linspace(0, 0.01, num)])
    savename = ""
    i1i2 = index_distributor(num, num)
    results = universal_runner(i1i2, paras, dest=False)

    global_results = comm.gather(results, root=0)
    if rank == 0:
        final_results = global_results[0]
        for k in range(1, len(global_results)):
            final_results = np.vstack((final_results, global_results[k]))
        np.save(savename, final_results)
    else:
        assert global_results is None

