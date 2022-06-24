import numpy as np
import efel
import multiprocessing as mlt
import matplotlib.pyplot as plt
import bluepyopt.ephys as ephys
import IVL_helper as ivlh
from neuron import h


h.load_file("nrngui.hoc")
h.celsius = 34
h.v_init = -74
h.dt = 0.1


def generate_full_cell():
    """Create the 1 compartment cell model with everything set, except the OU input parameters"""
    # Set up 1 compartment morphology
    morph = ephys.morphologies.NrnFileMorphology("m1053s.swc")

    # Set up somatic section_list for mechanisms to be inserted
    somatic_loc = ephys.locations.NrnSeclistLocation('somatic', seclist_name='somatic')

    # gbar params

    gbar_na_param = ephys.parameters.NrnSectionParameter(name='gna_Nasoma', param_name='gna_Nasoma',
                                                         value=70.986184915201491e-4, locations=[somatic_loc],
                                                         frozen=True)
    gbar_kdrf_param = ephys.parameters.NrnSectionParameter(name='gbar_Ikdrf', param_name='gbar_Ikdrf',
                                                           value=115.46932938891074e-4, locations=[somatic_loc],
                                                           frozen=True)
    gbar_m_param = ephys.parameters.NrnSectionParameter(name='gbar_IM', param_name='gbar_IM',
                                                        value=0.13738940328219354e-4, locations=[somatic_loc],
                                                        frozen=True)
    gbar_ka_param = ephys.parameters.NrnSectionParameter(name='gbar_Ika', param_name='gbar_Ika',
                                                         value=76.077776610698493e-4, locations=[somatic_loc],
                                                         frozen=True)
    gbar_h_param = ephys.parameters.NrnSectionParameter(name='gkhbar_Ih', param_name='gkhbar_Ih', value=1.06309e-5,
                                                        locations=[somatic_loc], frozen=True)


    # Create Mechanisms with mod files
    kdrf_mech = ephys.mechanisms.NrnMODMechanism(name='kdrf', mod_path='Ikdrf.mod', suffix='Ikdrf',
                                                 locations=[somatic_loc])
    na_mech = ephys.mechanisms.NrnMODMechanism(name='na', mod_path='Nasoma.mod', suffix='Nasoma',
                                               locations=[somatic_loc])
    m_mech = ephys.mechanisms.NrnMODMechanism(name='m', mod_path='IMmintau.mod', suffix='IM', locations=[somatic_loc])
    h_mech = ephys.mechanisms.NrnMODMechanism(name='h', mod_path='Ih.mod', suffix='Ih', locations=[somatic_loc])
    ka_mech = ephys.mechanisms.NrnMODMechanism(name='ka', mod_path='IKa.mod', suffix='Ika', locations=[somatic_loc])
    l_mech = ephys.mechanisms.NrnMODMechanism(name='l', mod_path='Ipasssd.mod', suffix='passsd',
                                              locations=[somatic_loc])

    # NaS Parameters
    vshift_na_param = ephys.parameters.NrnSectionParameter(name='vshift_na', param_name='vshift_Nasoma',
                                                           value=-4.830346371483079, locations=[somatic_loc],
                                                           frozen=True)



    # Leak Parameters
    gbar_l_param = ephys.parameters.NrnSectionParameter(name='gbar_l', param_name='g_passsd', value=7.5833e-06,
                                                        locations=[somatic_loc], frozen=True)
    erev_l_param = ephys.parameters.NrnSectionParameter(name='erev_passsd', param_name='erev_passsd', value=-64.640,
                                                        locations=[somatic_loc], frozen=True)

    # Ih parameters
    param_eh_h = ephys.parameters.NrnSectionParameter(name='eh_h', param_name='eh', value=-34.0056,
                                                      locations=[somatic_loc], frozen=True)
    param_v_half_h = ephys.parameters.NrnSectionParameter(name='v_half_h', param_name='v_half_Ih', value=-103.69,
                                                          locations=[somatic_loc], frozen=True)
    param_k_h = ephys.parameters.NrnSectionParameter(name='k_h', param_name='k_Ih', value=9.9995804,
                                                     locations=[somatic_loc], frozen=True)

    param_t1_h = ephys.parameters.NrnSectionParameter(name='t1_h', param_name='t1_Ih', value=8.5657797,
                                                      locations=[somatic_loc], frozen=True)
    param_t2_h = ephys.parameters.NrnSectionParameter(name='t2_h', param_name='t2_Ih', value=0.0296317,
                                                      locations=[somatic_loc], frozen=True)
    param_t3_h = ephys.parameters.NrnSectionParameter(name='t3_h', param_name='t3_Ih', value=-6.9145,
                                                      locations=[somatic_loc], frozen=True)
    param_t4_h = ephys.parameters.NrnSectionParameter(name='t4_h', param_name='t4_Ih', value=0.1803,
                                                      locations=[somatic_loc], frozen=True)
    param_t5_h = ephys.parameters.NrnSectionParameter(name='t5_h', param_name='t5_Ih', value=4.3566601e-05,
                                                      locations=[somatic_loc], frozen=True)

    # Global properties
    param_ena = ephys.parameters.NrnSectionParameter(name='ena', param_name='ena', value=90, locations=[somatic_loc],
                                                     frozen=True)
    param_ek = ephys.parameters.NrnSectionParameter(name='ek', param_name='ek', value=-95, locations=[somatic_loc],
                                                    frozen=True)
    param_cm = ephys.parameters.NrnSectionParameter(name='cm', param_name='cm', value=0.27008, locations=[somatic_loc],
                                                    frozen=True)

    # Instantiate cell
    m1053s = ephys.models.CellModel(name='m1053s', morph=morph,
                                    mechs=[kdrf_mech, na_mech, m_mech, h_mech, ka_mech, l_mech],
                                    params=[gbar_na_param, gbar_kdrf_param, gbar_ka_param, gbar_h_param, gbar_m_param,
                                            vshift_na_param, gbar_l_param, erev_l_param, param_eh_h,
                                            param_v_half_h, param_k_h, param_t1_h, param_t2_h, param_t3_h, param_t4_h,
                                            param_t5_h, param_ena, param_ek, param_cm])

    return m1053s


def init_ivl(cell_model, value_dict):
    # Object that points to the center of the soma
    somacenter_loc = ephys.locations.NrnSeclistCompLocation(
        name='somacenter',
        seclist_name='somatic',
        sec_index=0,
        comp_x=0.5)

    gfluct_mech = ephys.mechanisms.NrnMODPointProcessMechanism(name='gfluct', suffix='Gfluct2', mod_path='GFluct.mod',
                                                               locations=[somacenter_loc])

    gfluct_loc = ephys.locations.NrnPointProcessLocation('gfluct_loc', pprocess_mech=gfluct_mech)

    Ee = ephys.parameters.NrnPointProcessParameter('E_e', value=0, frozen=True, locations=[gfluct_loc],
                                                   param_name='E_e')
    Ei = ephys.parameters.NrnPointProcessParameter('E_i', value=-87.1, frozen=True,  locations=[gfluct_loc],
                                                   param_name='E_i')

    stde = ephys.parameters.NrnPointProcessParameter('std_e', frozen=False, bounds=None, locations=[gfluct_loc],
                                                     param_name='std_e')
    stdi = ephys.parameters.NrnPointProcessParameter('std_i', frozen=False, bounds=None, locations=[gfluct_loc],
                                                     param_name='std_i')

    ge0 = ephys.parameters.NrnPointProcessParameter('g_e0', frozen=False, bounds=None, locations=[gfluct_loc],
                                                    param_name='g_e0')
    gi0 = ephys.parameters.NrnPointProcessParameter('g_i0', frozen=False, bounds=None, locations=[gfluct_loc],
                                                    param_name='g_i0')

    taue = ephys.parameters.NrnPointProcessParameter('tau_e', frozen=False, bounds=None, locations=[gfluct_loc],
                                                     param_name='tau_e')
    taui = ephys.parameters.NrnPointProcessParameter('tau_i', frozen=False, bounds=None, locations=[gfluct_loc],
                                                     param_name='tau_i')

    cell_model.mechanisms.append(gfluct_mech)

    cell_model.params[Ee.name] = Ee
    cell_model.params[Ei.name] = Ei
    cell_model.params[stde.name] = stde
    cell_model.params[stdi.name] = stdi
    cell_model.params[ge0.name] = ge0
    cell_model.params[gi0.name] = gi0
    cell_model.params[taue.name] = taue
    cell_model.params[taui.name] = taui

    cell_model.freeze(value_dict)


def set_up_protocol():
    """Instantiate step and hold protocol"""
    soma_loc = ephys.locations.NrnSeclistCompLocation(name='soma', seclist_name='somatic', sec_index=0, comp_x=0.5)

    protocols = []
    for amp in [-0.12, 0.03, 0.06]:
        sim_name = 'Step' + str(amp)
        step = ephys.stimuli.NrnSquarePulse(step_amplitude=amp, step_delay=1000, step_duration=2000,
                                            total_duration=4000, location=soma_loc)
        hold = ephys.stimuli.NrnSquarePulse(step_amplitude=0.004, step_delay=0, step_duration=4000, total_duration=4000,
                                            location=soma_loc)
        rec = ephys.recordings.CompRecording(name=sim_name + '.soma.v', location=soma_loc, variable='v')
        step_protocol = ephys.protocols.SweepProtocol(name=sim_name, stimuli=[step, hold], recordings=[rec],
                                                      cvode_active=True)
        protocols.append(step_protocol)

    full_protocol = ephys.protocols.SequenceProtocol('three_steps', protocols=protocols)
    return full_protocol


def test_run():
    model = generate_full_cell()
    full_pro = set_up_protocol()
    nrn = ephys.simulators.NrnSimulator()
    responses = full_pro.run(model, {}, sim=nrn)
    keys = list(responses.keys())
    for i in range(len(keys)):
        plt.subplot(3, 1, i + 1)
        plt.plot(responses[keys[i]]['time'], responses[keys[i]]['voltage'], label=keys[i])
        plt.legend()
    plt.savefig('test_bluepyopt.png')


def test_run_ivl():
    gfluct_val = {'std_e': 0.003, 'std_i': 0.0066, 'tau_e': 12.7, 'tau_i': 12.05, 'g_e0': 0.01, 'g_i0': 0.03}
    model = generate_full_cell()
    init_ivl(model, gfluct_val)

    sim_name = 'Step'
    soma_loc = ephys.locations.NrnSeclistCompLocation(name='soma', seclist_name='somatic', sec_index=0, comp_x=0.5)
    hold = ephys.stimuli.NrnSquarePulse(step_amplitude=0, step_delay=0, step_duration=4000, total_duration=4000,
                                        location=soma_loc)
    rec = ephys.recordings.CompRecording(name=sim_name + '.soma.v', location=soma_loc, variable='v')
    step_protocol = ephys.protocols.SweepProtocol(name=sim_name, stimuli=[hold], recordings=[rec],
                                                  cvode_active=False)
    full_protocol = ephys.protocols.SequenceProtocol('drift', protocols=[step_protocol])

    nrn = ephys.simulators.NrnSimulator(cvode_active=False)
    responses = full_protocol.run(model, {}, sim=nrn)
    keys = list(responses.keys())
    for i in range(len(keys)):
        plt.subplot(3, 1, i + 1)
        plt.plot(responses[keys[i]]['time'], responses[keys[i]]['voltage'], label=keys[i])
        plt.legend()
    plt.savefig('test_ivl_bluepyopt.png')


# def multi_optimize():
#     processes = []
#     for gbar_param in FULL_GBAR_PARAMS:
#         opt_run = mlt.Process(target=single_optimize, args=(FULL_GBAR_PARAMS, FULL_GBAR_Bounds, [gbar_param.name]))
#         processes.append(opt_run)
#     for process in processes:
#         process.start()
#     for process in processes:
#         process.join()


def single_run(gfluct_val):
    g_value = np.linspace(0, 0.1, 50)
    std_value = np.linspace(0, 0.01, 50)
    model = generate_full_cell()

    pass


def multi_tau_ori_run():
    pass


if __name__ == "__main__":
    mlt.freeze_support()
    # test_run()
    test_run_ivl()
