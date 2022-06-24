from neuron import h
import record_1comp as r1
from IVL_helper import *
import multiprocessing as mp
import IVL_select
import numpy as np
import mpi4py.MPI as mpi

comm = mpi.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"this is node {rank} out of {size}\n")


def foo(num, dict):
    dict[num] = num


result_dict = mp.Manager().dict()
proc_list = []
for i in range(32):
    p = mp.Process(target=foo, args=(i + rank, result_dict))
    p.start()
    proc_list.append(p)
for p in proc_list:
    p.join()
local_result = []
for key in result_dict.keys():
    local_result.append(result_dict[key])

local_data = np.array(local_result)

global_data = comm.gather(local_data, root=0)

if rank == 0:
    global_data = np.array(global_data)
    np.save("test_mpi.npy", global_data)
else:
    print(local_data is None)
    print(global_data is None)
