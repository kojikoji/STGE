import numpy as np
from data_manupulation import data_manupulation
import json
from simulated_data_manager import simulated_data_manager
from STGE import STGE


stage_dict = json.load(open("data/base_data/stage_hpf.json"))
sim_dm = simulated_data_manager(100)
sim_dm.register_tomoseq(
    "data/base_data/tomo_seq/zfshield", stage_dict["shield"])
sim_dm.register_tomoseq(
    "data/base_data/tomo_seq/zf10ss", stage_dict["10ss"])
t_vec = np.array([10.9, 12.4, 14.4])
sim_dm.increase_sc_time_points(t_vec)
sim_dm.gen_simulation(10, 10)
sim_dm.process()
sim_stge = data_manupulation.initiate_sim_stge(sim_dm)
sim_stge = data_manupulation.optimize_stge(sim_stge)
