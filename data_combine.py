import numpy as np
import os
import os.path


def rec_combiner(path1, path2):
    for name in os.listdir(path1):
        if not os.path.isfile(f"{path1}/{name}"):
            rec_combiner(f"{path1}/{name}", f"{path2}/{name}")
        elif ".npy" in name:
            content1 = np.load(f"{path1}/{name}")
            content2 = np.load(f"{path2}/{name}")
            new_content = np.concatenate((content1, content2), axis=0)
            np.save(f"{path1}/{name}", new_content)


base_path1 = ""
base_path2 = ""
states = []


