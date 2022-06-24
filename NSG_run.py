from neuron import h
import record_1comp as r1
from IVL_helper import *
import multiprocessing as mp

# Instantiate Model
h.load_file("init_1comp.hoc")
h.cvode_active(0)
h.dt = 0.1
h.steps_per_ms = 10
DNQX_Hold = 0.004
TTX_Hold = -0.0290514
# cell = h.cell


def func1(i1, dic):
    g_value = np.linspace(0, 0.1, 50)
    std_value = np.linspace(0, 0.01, 50)

    runtime = 10000
    h.tstop = runtime

    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1

    # OU_in.tau_e = 12.7
    # OU_in.tau_i = 12.05

    v_hvec = r1.set_up_full_recording()[0]

    t_vec = np.array((range(int(runtime / h.dt) + 1))) * h.dt

    results = None

    for i2 in range(25):
        for j1 in range(50):
            for j2 in range(50):
                OU_in.g_e0 = g_value[i1]
                OU_in.g_i0 = g_value[i2]
                OU_in.std_e = std_value[j1]
                OU_in.std_i = std_value[j2]

                h.run()

                v_vec = np.array(v_hvec)
                score = evaluator3(t_vec, v_vec)
                if score == 4:
                    value_array = np.array([g_value[i1], g_value[i2], std_value[j1], std_value[j2]])
                    if results is None:
                        results = value_array
                    else:
                        results = np.vstack((results, value_array))

    dic[i1] = results


def func2(i1, dic):
    g_value = np.linspace(0, 0.1, 50)
    std_value = np.linspace(0, 0.01, 50)

    runtime = 10000
    h.tstop = runtime

    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1

    # OU_in.tau_e = 12.7
    # OU_in.tau_i = 12.05

    v_hvec = r1.set_up_full_recording()[0]

    t_vec = np.array((range(int(runtime / h.dt) + 1))) * h.dt

    results = None

    for i2 in range(25, 50):
        for j1 in range(50):
            for j2 in range(50):
                OU_in.g_e0 = g_value[i1]
                OU_in.g_i0 = g_value[i2]
                OU_in.std_e = std_value[j1]
                OU_in.std_i = std_value[j2]

                h.run()

                v_vec = np.array(v_hvec)
                score = evaluator3(t_vec, v_vec)
                if score == 4:
                    value_array = np.array([g_value[i1], g_value[i2], std_value[j1], std_value[j2]])
                    if results is None:
                        results = value_array
                    else:
                        results = np.vstack((results, value_array))

    dic[-i1 - 1] = results


def func3(i1, i2, dic):
    g_value = np.linspace(0, 0.1, 50)
    std_value = np.linspace(0, 0.01, 50)

    runtime = 10000
    h.tstop = runtime

    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1

    # OU_in.tau_e = 12.7
    # OU_in.tau_i = 12.05

    v_hvec = r1.set_up_full_recording()[0]

    t_vec = np.array((range(int(runtime / h.dt) + 1))) * h.dt

    results = None

    for j1 in range(50):
        for j2 in range(50):
            OU_in.g_e0 = g_value[i1]
            OU_in.g_i0 = g_value[i2]
            OU_in.std_e = std_value[j1]
            OU_in.std_i = std_value[j2]

            h.run()

            v_vec = np.array(v_hvec)
            score = evaluator3(t_vec, v_vec)
            if score == 4:
                value_array = np.array([g_value[i1], g_value[i2], std_value[j1], std_value[j2]])
                if results is None:
                    results = value_array
                else:
                    results = np.vstack((results, value_array))

    dic[(i1, i2)] = results


def func4(i1, i2, g_array, std_array, dic, dest):
    g_value = g_array
    std_value = std_array

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

    for j1 in range(50):
        for j2 in range(50):
            OU_in.g_e0 = g_value[i1]
            OU_in.g_i0 = g_value[i2]
            OU_in.std_e = std_value[j1]
            OU_in.std_i = std_value[j2]

            h.run()

            v_vec = np.array(v_hvec)
            score = evaluator3(t_vec, v_vec)
            if score == 4:
                value_array = np.array([g_value[i1], g_value[i2], std_value[j1], std_value[j2]])
                if results is None:
                    results = value_array
                else:
                    results = np.vstack((results, value_array))

    dic[(i1, i2)] = results


def partial_func1(i1, i2, ge_array, gi_array, stde_array, stdi_array, dic, dest):
    """ge0 = 0.01; gi0=0.03-0.04, stde=0.0005-0.001, stdi=0.009-0.01"""

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

    for j1 in range(stde_array.shape[0]):
        for j2 in range(stdi_array.shape[0]):
            OU_in.g_e0 = ge_array[i1]
            OU_in.g_i0 = gi_array[i2]
            OU_in.std_e = stde_array[j1]
            OU_in.std_i = stdi_array[j2]

            h.run()

            v_vec = np.array(v_hvec)
            score = evaluator3(t_vec, v_vec)
            if score == 4:
                value_array = np.array([ge_array[i1], gi_array[i2], stde_array[j1], stdi_array[j2]])
                if results is None:
                    results = value_array
                else:
                    results = np.vstack((results, value_array))

    dic[(i1, i2)] = results


def single_tau_search(val_array, dic):
    tau_value = np.linspace(0, 25, 50)

    runtime = 10000
    h.tstop = runtime

    OU_in = h.Gfluct2(h.soma(0.5))
    OU_in.E_i = -87.1

    v_hvec = r1.set_up_full_recording()[0]

    t_vec = np.array((range(int(runtime / h.dt) + 1))) * h.dt

    results = None
    OU_in.g_e0 = val_array[0]
    OU_in.g_i0 = val_array[1]
    OU_in.std_e = val_array[2]
    OU_in.std_i = val_array[3]
    for i in range(50):
        for j in range(50):
            OU_in.tau_e = tau_value[i]
            OU_in.tau_i = tau_value[j]

            h.run()

            v_vec = np.array(v_hvec)
            score = evaluator3(t_vec, v_vec)
            if score == 4:
                value_array = np.array([val_array[0], val_array[1], val_array[2],
                                        val_array[3], tau_value[i], tau_value[j]])
                if results is None:
                    results = value_array
                else:
                    results = np.vstack((results, value_array))

    dic[(val_array[0], val_array[1], val_array[2], val_array[3])] = results


def run1(save_path, num):
    proc_list = []
    result_dict = mp.Manager().dict()
    for i1 in range(num):
        p1 = mp.Process(target=func1, args=(i1, result_dict))
        p2 = mp.Process(target=func2, args=(i1, result_dict))
        proc_list.append(p1)
        proc_list.append(p2)

    for p in proc_list:
        p.start()
    for p in proc_list:
        p.join()

    results = None
    for key in list(result_dict):
        if results is None:
            results = result_dict[key]
        else:
            results = np.vstack((results, result_dict[key]))

    np.save(save_path, results)


def run2(save_path, num):
    proc_list = []
    result_dict = mp.Manager().dict()
    for i1 in range(num):
        for i2 in range(num):
            p1 = mp.Process(target=func3, args=(i1, i2, result_dict))
            proc_list.append(p1)

    for p in proc_list:
        p.start()
    for p in proc_list:
        p.join()

    results = None
    for key in list(result_dict):
        if result_dict[key] is None:
            sing_result = np.array([-1, -1, -1, -1])
        else:
            sing_result = result_dict[key]

        if results is None:
            results = sing_result
        else:
            results = np.vstack((results, sing_result))

    np.save(save_path, results)


def run3(save_path, num, dest):
    proc_list = []
    result_dict = mp.Manager().dict()
    g_value = np.linspace(0, 0.5, 50)       # 0 to 0.5; change from 0 to 0.1
    # std_value = np.linspace(0, 0.01, 50)
    std_value = np.linspace(0, 0.1, 50)
    for i1 in range(num):
        for i2 in range(num):
            p1 = mp.Process(target=func4, args=(i1, i2, g_value, std_value, result_dict, dest))
            proc_list.append(p1)

    for p in proc_list:
        p.start()
    for p in proc_list:
        p.join()

    results = None
    for key in list(result_dict):
        if result_dict[key] is None:
            sing_result = np.array([-1, -1, -1, -1])
        else:
            sing_result = result_dict[key]

        if results is None:
            results = sing_result
        else:
            results = np.vstack((results, sing_result))

    results = results[results[:, -1] != -1]

    np.save(save_path, results)


def partial_run1(save_path, num, dest):
    """ge0 = 0.01; gi0=0.03-0.04, stde=0.0005-0.001, stdi=0.009-0.01"""
    proc_list = []
    result_dict = mp.Manager().dict()
    ge_value = [0.01]  # 0 to 0.5; change from 0 to 0.1
    gi_value = np.linspace(0.03, 0.04, 10)
    stde_value = np.linspace(0.0005, 0.001, 10)
    stdi_value = np.linspace(0.009, 0.01, 10)
    for i2 in range(num):
        p1 = mp.Process(target=partial_func1, args=(0, i2, ge_value, gi_value, stde_value, stdi_value, result_dict, dest))
        proc_list.append(p1)

    for p in proc_list:
        p.start()
    for p in proc_list:
        p.join()

    results = None
    for key in list(result_dict):
        if result_dict[key] is None:
            sing_result = np.array([-1, -1, -1, -1])
        else:
            sing_result = result_dict[key]

        if results is None:
            results = sing_result
        else:
            results = np.vstack((results, sing_result))

    results = results[results[:, -1] != -1]

    np.save(save_path, results)


def multi_tau_search(para_path, save_path):
    paras = np.load(para_path)
    result_dict = mp.Manager().dict()
    proc_list = []
    for i in range(paras.shape[0]):
        p = mp.Process(target=single_tau_search, args=(paras[i], result_dict))
        p.start()
        proc_list.append(p)

    for p in proc_list:
        p.join()

    results = None
    for key in list(result_dict):
        if results is None:
            results = result_dict[key]
        else:
            results = np.vstack((results, result_dict[key]))

    np.save(save_path, results)


if __name__ == "__main__":
    mp.freeze_support()
    # run1("full_search_1.npy", 50)
    # run2("full_search_1.npy", 50)
    # run3("full_search_2_dest.npy", 50, dest=True)
    # run3("full_search_2_agm.npy", 50, dest=False)
    # run3("full_search_3_dest.npy", 50, dest=True)
    # run3("full_search_3_agm.npy", 50, dest=False)
    partial_run1('partial_search_1_agm_opt.npy', 10, dest=False)
