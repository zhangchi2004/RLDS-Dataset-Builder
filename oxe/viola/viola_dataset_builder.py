import os
import tensorflow_datasets as tfds
from typing import Iterator, Dict, Any, Tuple
import glob
import pickle
import numpy as np

SRC_PATH = "/home/projects/xlang.slurm/czhang/data/OXE/viola"  # Parent dir containing subdirs (each subdir = dataset)

class Viola(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Initializing Viola dataset builder...")
        self._feature_structure = None

    def _infer_feature(self, value: Any) -> Any:
        # import pdb; pdb.set_trace()  # Debugging line to inspect value
        """Infers TFDS feature type from Python value with strict typing"""
        if isinstance(value, np.ndarray):
            return tfds.features.Tensor(shape=value.shape, dtype=value.dtype)
        elif isinstance(value, (int, np.integer)):
            return tfds.features.Scalar(dtype=type(value))
        elif isinstance(value, (float, np.floating)):
            return tfds.features.Scalar(dtype=type(value))
        elif isinstance(value, (bool, np.bool_)):
            print(value, "is bool")
            return tfds.features.Scalar(dtype=np.bool_)
        elif isinstance(value, str):
            print(value, "is str")
            return tfds.features.Text()
        elif isinstance(value, dict):
            return tfds.features.FeaturesDict({
                k: self._infer_feature(v) for k, v in value.items()
            })
        elif isinstance(value, (list, tuple)):
            if len(value) > 0:
                return tfds.features.Sequence(self._infer_feature(value[0]))
            return tfds.features.Sequence(tfds.features.Tensor(shape=(), dtype=np.object_))
        else:
            return tfds.features.Tensor(shape=(), dtype=np.object_)

    def _info(self) -> tfds.core.DatasetInfo:
        """Strictly mirrors pickle structure with no added fields"""
        self._feature_structure = None
        if self._feature_structure is None:
            # Find first pickle to infer structure
            pickle_files = glob.glob(f"{SRC_PATH}/*.pickle")
            if not pickle_files:
                raise ValueError("No pickle files found")
            
            with open(pickle_files[0], "rb") as f:
                sample_data = pickle.load(f)
            self._feature_structure = self._infer_feature(sample_data)
        
        print("Feature structure inferred:", self._feature_structure)
        return self.dataset_info_from_configs(
            features=self._feature_structure
        )
    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples(SRC_PATH)}

    def _generate_examples(self, dir_path) -> Iterator[Tuple[str, Any]]:
        print(dir_path)
        # Your existing pickle-loading logic here, but for ONE directory
        pickle_files = sorted(glob.glob(f"{dir_path}/*.pickle"))
        
        for step_idx, pickle_file in enumerate(pickle_files):
            with open(pickle_file, "rb") as f:
                data = pickle.load(f)
            # import pdb; pdb.set_trace()  # Debugging line to inspect data
            yield str(step_idx), data