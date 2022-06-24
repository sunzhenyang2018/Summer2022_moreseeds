import IVL_plots as ivp
import numpy as np
import mpi4py.MPI as mpi
import multiprocessing as mp

comm = mpi.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

gstb_path = "ofs2_gstb.npy"
paras = np.load(gstb_path)
start = rank * int(paras.shape[0] // size)

if rank == size - 1:
    paras = paras[start: ]
else:
    end = (rank + 1) * int(paras.shape[0] // size)
    paras = paras[start: end]


def rate_from_binned(runtime, num_div, paras, savepath, save=True):
    # [bin_num, ge, gi, stde, stdi, ...]
    slice_size = int(paras.shape[0] // num_div)
    proc_list = []
    shared_dic = mp.Manager().dict()
    for i in range(0, paras.shape[0], slice_size):
        p = mp.Process(target=ivp.single_rate_from_binned, args=(paras[i: i + slice_size], runtime, shared_dic, i))
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


local_results = rate_from_binned(10000, 128, paras, "", save=False)
all_results = comm.gather(local_results, root=0)
if rank == 0:
    final_result = None
    for result in all_results:
        if final_result is None:
            final_result = result
        else:
            final_result = np.vstack((final_result, result))
    if final_result is None:
        print("Mission failed, we'll get them next time.")
    else:
        np.save("ofs2_gstdbr.npy", final_result)
else:
    pass
