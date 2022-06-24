import numpy
from neuron import h, gui
import record_1comp as r1
import efel
import matplotlib.pyplot as plt
from currents_visualization import *

### Instantiate Model ###
h.load_file("init_1comp.hoc")
h.cvode_active(0)
h.dt = 0.1
h.steps_per_ms = 10
DNQX_Hold = 0.004
TTX_Hold = -0.0290514


def voltage_wt_perturbation(hold, pert_amp, pert_freq, time):
    h.ic_hold.amp = hold
    h.ic_hold.delay = 100
    h.ic_hold.dur = time
    h.tstop = time


    inter_pert_unit = int(1000 // pert_freq)
    pert_vector = np.zeros(int(time))
    for i in range(150, len(pert_vector), inter_pert_unit):
        pert_vector[i] = pert_amp

    ipert = h.IClamp(h.soma(0.5))
    ipert.delay=0
    ipert.dur=1e9
    timeVec = h.Vector(range(int(time)))
    a = h.Vector(pert_vector)

    a.play(ipert._ref_amp, timeVec, 1)  # The meaning of dt: dt of pert-vector compared to time vector
    b = np.array(a)
    print(sum(b))
    vecs = r1.record_soma()
    h.run()
    vecs = [np.array(i) for i in vecs]
    plt.plot(vecs[0])
    plt.savefig("1_comp_plots/pert/hold"+str(hold)+"_pert"+str(pert_amp)+"_freq"+str(pert_freq)+".png", dpi=150)
    # plt.show()


def fit_frequency(hold, time):
    h.ic_hold.amp = hold
    h.ic_hold.dur = time
    h.tstop = time

    efel.api.setThreshold(-20)
    efel.api.setDerivativeThreshold(1)

    vecs = r1.set_up_full_recording()
    h.run()
    v_trace = np.array(vecs[0])
    tvec = numpy.arange(0, len(v_trace), 1)
    tvec = tvec * h.dt                      # mul dt to get 1/1000 second, efel also uses ms
    trace = {'V': v_trace, 'T': tvec, 'stim_start': [0]}

    trace['stim_end'] = [time]

    ef_list = ['mean_frequency']
    traces_result = efel.getFeatureValues([trace], ef_list)

    plt.plot(tvec, v_trace)
    plt.show()
    return traces_result


def is3_pert_run(hold, pert_amp, pert_freq, time):
    h.ic_hold.amp = hold
    h.ic_hold.delay = 0
    h.ic_hold.dur = time
    h.tstop = time

    stim = h.NetStim()
    stim.interval = 1/pert_freq * 1000       # ms mean time between spikes
    stim.number = int((time-250)*h.dt*1000*pert_freq)        # average number of spikes. convert to ms, then s
    stim.start = 250          # ms start of stim
    stim.noise = 0          # deterministic

    syn = h.Exp2Syn(h.soma(0.5))
    syn.tau1 = 1.6          # ms rise time
    syn.tau2 = 12.0         # ms decay time
    syn.e = -87.1           # reversal potential

    netcon = h.NetCon(stim, syn)    # threshold is irrelevant with event-based source
    netcon.weight[0] = pert_amp
    vecs = r1.record_soma()
    h.run()
    vecs = [np.array(i) for i in vecs]

    plt.plot(vecs[0][1800:])
    plt.savefig("1_comp_plots/pert/hold" + str(hold) + "_pert" + str(pert_amp) + "_freq" + str(pert_freq) + ".png",
                dpi=150)
    # plt.show()


def plot_frequency():
    run_time = 10000
    h.tstop = run_time
    vecs = r1.set_up_full_recording()
    h.ic_hold.dur = run_time
    h.ic_hold.delay = 0
    frequencies = []

    efel.api.setThreshold(-20)
    efel.api.setDerivativeThreshold(1)

    for i in range(36):
        h.ic_hold.amp = i * 0.001   # 1 pA = 0.001
        h.run()
        v_trace = np.array(vecs[0])
        tvec = numpy.arange(0, len(v_trace), 1)
        tvec = tvec * h.dt          # mul dt to get 1/1000 second, efel also uses ms
        trace = {'V': v_trace, 'T': tvec, 'stim_start': [0], 'stim_end': [run_time]}
        ef_list = ['mean_frequency']
        traces_result = efel.getFeatureValues([trace], ef_list)
        if traces_result[0]['mean_frequency'] is None:
            frequencies.append(0)
        else:
            frequencies.append(traces_result[0]['mean_frequency'][0])
    plt.scatter(list(range(36)), frequencies)
    plt.savefig('1_comp_plots/frequencies.png')


def ISI_finder_vitro(hold, time):
    h.ic_hold.amp = hold
    h.ic_hold.delay = 0
    h.ic_hold.dur = time
    h.tstop = time

    vecs = r1.record_soma()
    h.run()

    efel.api.setThreshold(-20)
    efel.api.setDerivativeThreshold(1)

    v_trace = np.array(vecs[0])
    tvec = numpy.arange(0, len(v_trace), 1)
    tvec = tvec * h.dt # mul dt to get 1/1000 second, efel also uses ms
    trace = {'V': v_trace, 'T': tvec, 'stim_start': [0], 'stim_end': [time]}

    ef_list = ['peak_time']
    spike_times = efel.getFeatureValues([trace], ef_list)
    print(spike_times)
    plt.plot(v_trace)
    plt.show()


def currentscape_wt_pert(hold_amp, runtime):
    # set up spiking neuron running 1 second
    h.ic_hold.amp = hold_amp
    h.ic_hold.delay = 0
    h.ic_hold.dur = runtime
    h.tstop = runtime

    vecs = r1.set_up_full_recording()
    v_vec = vecs[0]

    # find time for 4th and 5th spike without perturbation, determine t0
    h.run()
    v_trace = np.array(v_vec)
    ef_list = ['peak_time']
    tvec = np.array(range(int(runtime / h.dt) + 1))
    tvec = tvec * h.dt  # mul dt to get 1/1000 second, efel also uses ms
    trace = {'V': v_trace, 'T': tvec, 'stim_start': [0], 'stim_end': [runtime]}
    spike_times = efel.getFeatureValues([trace], ef_list)
    spike3 = spike_times[0]['peak_time'][2]
    spike4 = spike_times[0]['peak_time'][3]
    spike5 = spike_times[0]['peak_time'][4]
    t0 = spike5 - spike4  # Alex's implementation would be spike4 - spike3

    # creating synapse
    stim = h.NetStim()
    stim.number = 1  # average number of spikes. convert to ms, then s
    stim.noise = 0  # deterministic

    syn = h.Exp2Syn(h.soma(0.5))
    syn.tau1 = 1.6  # ms rise time
    syn.tau2 = 12.0  # ms decay time
    syn.e = -87.1  # reversal potential

    netcon = h.NetCon(stim, syn)  # threshold is irrelevant with event-based source
    netcon.weight[0] = 0.0054

    # perturbation at fourth ISI, taking the third as standard
    # the fourth spike normally start at 457.6 and end on 597.7
    percents = np.array(range(101))
    percents = percents * 0.01
    pert_time = percents * t0 + spike4

    for i in range(0, len(pert_time), 20):
        stim.start = pert_time[i]
        h.run()

        plotCurrentscape_6_current(np.array(vecs[0]), np.array(vecs[1:]))
        plt.savefig('1_comp_plots/currscape/' + str(i) + '.png')
        plt.close()


def PRC_vitro(hold_amp, runtime, prcloc, scatterloc):
    # set up spiking neuron running 1 second
    h.ic_hold.amp = hold_amp
    h.ic_hold.delay = 0
    h.ic_hold.dur = runtime
    h.tstop = runtime

    v_vec = r1.set_up_full_recording()[0]

    # find time for 4th and 5th spike without perturbation, determine t0
    h.run()
    v_trace = np.array(v_vec)
    ef_list = ['peak_time']
    tvec = np.array(range(int(runtime/h.dt) + 1))
    tvec = tvec * h.dt  # mul dt to get 1/1000 second, efel also uses ms
    trace = {'V': v_trace, 'T': tvec, 'stim_start': [0], 'stim_end': [runtime]}
    spike_times = efel.getFeatureValues([trace], ef_list)
    spike3 = spike_times[0]['peak_time'][2]
    spike4 = spike_times[0]['peak_time'][3]
    spike5 = spike_times[0]['peak_time'][4]
    t0 = spike5 - spike4        # Alex's implementation would be spike4 - spike3

    # creating synapse
    stim = h.NetStim()
    stim.number = 1  # average number of spikes. convert to ms, then s
    stim.noise = 0  # deterministic

    syn = h.Exp2Syn(h.soma(0.5))
    syn.tau1 = 1.6  # ms rise time
    syn.tau2 = 12.0  # ms decay time
    syn.e = -87.1  # reversal potential

    netcon = h.NetCon(stim, syn)  # threshold is irrelevant with event-based source
    netcon.weight[0] = 0.0054

    # perturbation at fourth ISI, taking the third as standard
    # the fourth spike normally start at 457.6 and end on 597.7
    percents = np.array(range(101))
    percents = percents * 0.01
    pert_time = percents * t0 + spike4

    phase_shifts = []
    for start_time in pert_time:
        stim.start = start_time
        h.run()
        v_trace = np.array(v_vec)
        trace = {'V': v_trace, 'T': tvec, 'stim_start': [0], 'stim_end': [runtime]}
        # plt.plot(v_trace)
        # plt.show()
        spike_times = efel.getFeatureValues([trace], ef_list)

        t1 = spike_times[0]['peak_time'][4] - spike4
        percent_shift = (t1-t0)/t0 * 100
        phase_shifts.append(percent_shift)

    plt.plot(percents, phase_shifts)
    plt.savefig(prcloc)     # '1_comp_plots/PRC-vitro.png'
    plt.close()
    plt.scatter(percents, phase_shifts)
    plt.savefig(scatterloc)           # '1_comp_plots/PRC-vitro-scatter.png'
    plt.close()


def _spike_time_finder(hold_amp, runtime):
    # set up spiking neuron running 1 second
    h.ic_hold.amp = hold_amp
    h.ic_hold.delay = 0
    h.ic_hold.dur = runtime
    h.tstop = runtime

    v_vec = r1.set_up_full_recording()[0]
    c_vecs = r1.set_up_full_recording()[1:]
    # find time for 4th and 5th spike without perturbation, determine t0
    h.run()
    v_trace = np.array(v_vec)
    ef_list = ['peak_time']
    tvec = np.array(range(int(runtime / h.dt) + 1))
    tvec = tvec * h.dt  # mul dt to get 1/1000 second, efel also uses ms
    trace = {'V': v_trace, 'T': tvec, 'stim_start': [0], 'stim_end': [runtime]}
    spike_times = efel.getFeatureValues([trace], ef_list)
    return spike_times, c_vecs


def _setting_stim():
    # creating synapse
    stim = h.NetStim()
    stim.number = 1  # average number of spikes. convert to ms, then s
    stim.noise = 0  # deterministic

    syn = h.Exp2Syn(h.soma(0.5))
    syn.tau1 = 1.6  # ms rise time
    syn.tau2 = 12.0  # ms decay time
    syn.e = -87.1  # reversal potential

    netcon = h.NetCon(stim, syn)  # threshold is irrelevant with event-based source
    netcon.weight[0] = 0.0054
    return stim


def PRCC_vitro(hold_amp, runtime, prcloc, curr_nums):  # PRC applied to currents
    spike_times, c_vecs = _spike_time_finder(hold_amp, runtime)
    spike3 = spike_times[0]['peak_time'][2]
    spike4 = spike_times[0]['peak_time'][3]
    spike5 = spike_times[0]['peak_time'][4]
    spike6 = spike_times[0]['peak_time'][5]

    stim = h.NetStim()
    stim.number = 1  # average number of spikes. convert to ms, then s
    stim.noise = 0  # deterministic

    syn = h.Exp2Syn(h.soma(0.5))
    syn.tau1 = 1.6  # ms rise time
    syn.tau2 = 12.0  # ms decay time
    syn.e = -87.1  # reversal potential

    netcon = h.NetCon(stim, syn)  # threshold is irrelevant with event-based source
    netcon.weight[0] = 0.0054

    # perturbation at fourth ISI, taking the third as standard
    # the fourth spike normally start at 457.6 and end on 597.7
    percents = np.array(range(101))
    percents = percents * 0.01
    pert_time = percents * (spike5 - spike4) + spike4

    phase_shifts = None
    for start_time in pert_time:
        stim.start = start_time
        h.run()
        cur_vecs = np.array([np.array(i) for i in c_vecs])
        # I0 is peak amp from 2nd last spike before pert to pert
        # I1 is peak amp from pert to 2nd spike after pert
        I0_vecs = np.max(abs(cur_vecs[:, int(spike3 / h.dt): int(start_time / h.dt)]), axis=1)  # nx1 array, always
        I1_vecs = np.max(abs(cur_vecs[:, int(start_time / h.dt): int(spike6 / h.dt)]), axis=1)  # nx1 array, always

        single_shift = (I1_vecs-I0_vecs) / I0_vecs * 100

        if phase_shifts is None:
            phase_shifts = single_shift
        else:
            # in the end, phase_shifts will be size: 100 x num of curs. 1st col is 1st current
            phase_shifts = np.vstack((phase_shifts, single_shift))

    # plt.plot(np.tile(np.array(range(101)), (phase_shifts.shape[1], 1)), phase_shifts.T)
    labels = ['ka', 'kdrf', 'm', 'l', 'na', 'h']
    for i in curr_nums:
        plt.plot(np.array(range(101)), phase_shifts[:, i], label=labels[i])
    plt.legend()
    plt.savefig(prcloc)     # '1_comp_plots/PRCC-vitro.png'
    plt.close()


def pyr_pert_run(hold, pert_amp, pert_freq, time):
    h.ic_hold.amp = hold
    h.ic_hold.delay = 0
    h.ic_hold.dur = time
    h.tstop = time

    stim = h.NetStim()
    stim.interval = 1 / pert_freq * 1000  # ms mean time between spikes
    stim.number = int((time) * h.dt * 1000 * pert_freq)  # average number of spikes. convert to ms, then s
    stim.start = 0 # ms start of stim
    stim.noise = 0  # deterministic

    syn = h.Exp2Syn(h.soma(0.5))
    syn.tau1 = 2.4  # ms rise time
    syn.tau2 = 12.7  # ms decay time
    syn.e = 0  # reversal potential from f1000

    netcon = h.NetCon(stim, syn)  # threshold is irrelevant with event-based source
    netcon.weight[0] = pert_amp
    vecs = r1.record_soma()
    h.run()
    vecs = [np.array(i) for i in vecs]

    plt.plot(vecs[0][90000:])
    plt.savefig("1_comp_plots/pert/PYR_hold" + str(hold) + "_pert" + str(pert_amp) + "_freq" + str(pert_freq) + ".png",
                dpi=150)
    # plt.show()


def txt_runs():
    for amp in [0.03, 0.06, 0.09, -0.12]:
        h.ic_step.amp = amp
        h.ic_hold.amp = DNQX_Hold

        v_vec = r1.set_up_full_recording()[0]
        h.run()
        v_vec = np.array(v_vec)
        np.savetxt(str(amp) + "pA.txt", v_vec)


# plot_frequency()
# print(fit_frequency(0.025425, 10000)) # 1 hz (3 sig fig) firing needed current inject
# PRC_vitro(0.025425, 10000, '1_comp_plots/PRC-vitro.png', '1_comp_plots/PRC-vitro-scatter.png')

# PRCC_vitro(0.025425, 10000, '1_comp_plots/PRCC-vitro1hz.png')
# currentscape_wt_pert(0.025425, 10000)

# print(fit_frequency(0.1568, 10000))  # 36 hz (3 sig fig) firing needed current inject
# print(fit_frequency(0.03508, 10000)) # 4 Hz
# print(fit_frequency(0.047395, 10000)) # 7.25 Hz
# print(fit_frequency(0.07565, 10000)) # 15 Hz

# PRC_vitro(0.03508, 10000, '1_comp_plots/PRC-vitro_4hz.png', '1_comp_plots/PRC-vitro_4hz-scatter.png')
# PRCC_vitro(0.03508, 10000, '1_comp_plots/PRCC-vitro_4hz.png', list(range(6)))
#
# PRC_vitro(0.047395, 10000, '1_comp_plots/PRC-vitro_7_25hz.png', '1_comp_plots/PRC-vitro_7_25hz-scatter.png')
# PRCC_vitro(0.047395, 10000, '1_comp_plots/PRCC-vitro_7_25hz.png', list(range(6)))
#
# PRC_vitro(0.07565, 10000, '1_comp_plots/PRC-vitro_15hz.png', '1_comp_plots/PRC-vitro_15hz-scatter.png')
# PRCC_vitro(0.07565, 10000, '1_comp_plots/PRCC-vitro_15hz.png', list(range(6)))

PRCC_vitro(0.025425, 10000, '1_comp_plots/PRCC-IM_1hz.png', [2])
PRCC_vitro(0.03508, 10000, '1_comp_plots/PRCC-IM_4hz.png', [2])
PRCC_vitro(0.047395, 10000, '1_comp_plots/PRCC-IM_7_25hz.png', [2])
PRCC_vitro(0.07565, 10000, '1_comp_plots/PRCC-IM_15hz.png', [2])

PRCC_vitro(0.025425, 10000, '1_comp_plots/PRCC-Ih_1hz.png', [5])
PRCC_vitro(0.03508, 10000, '1_comp_plots/PRCC-Ih_4hz.png', [5])
PRCC_vitro(0.047395, 10000, '1_comp_plots/PRCC-Ih_7_25hz.png', [5])
PRCC_vitro(0.07565, 10000, '1_comp_plots/PRCC-Ih_15hz.png', [5])
