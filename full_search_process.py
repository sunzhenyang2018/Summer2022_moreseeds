from IVL_search import *

g_binner("data/IVL_search/full_search_4_agm.npy", "data/IVL_search/fs4_agm_gb.npy")
g_binner("data/IVL_search/full_search_4_dest.npy", "data/IVL_search/fs4_dest_gb.npy")

g_binned_sort("data/IVL_search/fs4_agm_gb.npy", "data/IVL_search/fs4_agm_gb_sorted.npy")
g_binned_sort("data/IVL_search/fs4_dest_gb.npy", "data/IVL_search/fs4_dest_gb_sorted.npy")

plot_gs("data/IVL_search/fs4_agm_gb_sorted.npy", "1_comp_plots/IVL/paraplots/full_search_4_agm.png")
plot_gs("data/IVL_search/fs4_dest_gb_sorted.npy", "1_comp_plots/IVL/paraplots/full_search_4_dest.png")

stds_binner("data/IVL_search/fs4_agm_gb_sorted.npy", "data/IVL_search/full_search_4_agm.npy",
            "data/IVL_search/fs4_agm_gstdb.npy")
stds_binner(g_path="data/IVL_search/fs4_dest_gb_sorted.npy", g_std_path="data/IVL_search/full_search_4_dest.npy",
            savepath="data/IVL_search/fs4_dest_gstdb.npy")

plot_stds(gs_path="data/IVL_search/fs4_agm_gb_sorted.npy", g_std_binned_path="data/IVL_search/fs4_agm_gstdb.npy",
          savepath="1_comp_plots/IVL/paraplots/fs4_stds_agm.png")
plot_stds(gs_path="data/IVL_search/fs4_dest_gb_sorted.npy", g_std_binned_path="data/IVL_search/fs4_dest_gstdb.npy",
          savepath="1_comp_plots/IVL/paraplots/fs4_stds_dest.png")

plot_stds_no_overlap(division=135, gs_path="data/IVL_search/fs4_agm_gb_sorted.npy",
                     g_std_binned_path="data/IVL_search/fs4_agm_gstdb.npy",
                     base_savepath="1_comp_plots/IVL/paraplots/fs4_stds_agm.png")
plot_stds_no_overlap(division=75, gs_path="data/IVL_search/fs4_dest_gb_sorted.npy",
                     g_std_binned_path="data/IVL_search/fs4_dest_gstdb.npy",
                     base_savepath="1_comp_plots/IVL/paraplots/fs4_stds_dest.png")
