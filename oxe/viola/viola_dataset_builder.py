import os
import tensorflow_datasets as tfds
from typing import Iterator, Dict, Any, Tuple
import glob
import pickle

SRC_PATH = "/home/projects/xlang.slurm/czhang/data/OXE/viola"  # Parent dir containing subdirs (each subdir = dataset)

class Viola(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({}), # Empty features dict
        )

    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples(SRC_PATH)}

    def _generate_examples(self, dir_path) -> Iterator[Tuple[str, Any]]:
        print(dir_path)
        # Your existing pickle-loading logic here, but for ONE directory
        pickle_files = sorted(glob.glob(f"{dir_path}/*.pickle"))
        print(f"Found {len(pickle_files)} pickle files in {dir_path}")
        for step_idx, pickle_file in enumerate(pickle_files):
            with open(pickle_file, "rb") as f:
                data = pickle.load(f)
            yield str(step_idx), data