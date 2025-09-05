from typing import Iterator, Tuple, Any

import glob
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

import h5py
try:
    import apache_beam as beam
except ImportError:
    beam = None

# By zc

SRC_PATH = '/home2/czhang/datasets/Maniskill' # Modify this path to your dataset location

    

class Maniskill(tfds.core.GeneratorBasedBuilder): # Modify the class name to your dataset name
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot EEF action.',
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'observation': tfds.features.FeaturesDict({
                        'left_camera': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Left camera RGB observation.',
                        ),
                        'right_camera': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Right camera RGB observation.',
                        ),
                        'hand_camera': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB observation.',
                        ),
                        'joint_state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint angles.',
                        ),
                        'gripper_state': tfds.features.Tensor(
                            shape=(2,),
                            dtype=np.float32,
                            doc='Gripper State.'
                        )
                    }),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            # 'train': self._generate_examples(path=f'/home/czhang/ManiSkill/demos/03b2d88d-92b0-41f5-9754-812354706d80/motionplanning/20250825_210021.h5'),
            'train': self._generate_examples(path=f'/home/projects/xlang.slurm/czhang/20250825_210021.h5'),
            #                                  [
            #     f'{SRC_PATH}/LIVING_ROOM_SCENE2_place_the_alphabet_soup_in_the_basket_demo.hdf5',
            #     f'{SRC_PATH}/LIVING_ROOM_SCENE2_place_the_butter_in_the_basket_demo.hdf5',
            #     f'{SRC_PATH}/LIVING_ROOM_SCENE2_place_the_cream_cheese_in_the_basket_demo.hdf5',
            #     f'{SRC_PATH}/LIVING_ROOM_SCENE2_place_the_ketchup_in_the_basket_demo.hdf5',
            #     f'{SRC_PATH}/LIVING_ROOM_SCENE2_place_the_orange_juice_in_the_basket_demo.hdf5',
            #     f'{SRC_PATH}/LIVING_ROOM_SCENE2_place_the_tomato_sauce_in_the_basket_demo.hdf5',
            # ]),
            # 'val': self._generate_examples(path=f'{SRC_PATH}/LIVING_ROOM_SCENE2_place_the_milk_in_the_basket_demo.hdf5'),  # Modify this if you have a separate validation set
        }

    def _discover_h5_files(self, path):
        """Discover all H5/HDF5 files from path (supports directories, globs, and file lists)."""
        episode_paths = []
        
        if isinstance(path, str):
            # Check if it's a single file
            if os.path.isfile(path) and path.lower().endswith(('.h5', '.hdf5')):
                episode_paths = [path]
            # Check if it's a directory - scan recursively
            elif os.path.isdir(path):
                print(f"Scanning directory recursively: {path}")
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.lower().endswith(('.h5', '.hdf5')):
                            full_path = os.path.join(root, file)
                            episode_paths.append(full_path)
                print(f"Found {len(episode_paths)} H5 files")
            # Otherwise treat as glob pattern
            else:
                episode_paths = glob.glob(path)
                # Also try recursive glob patterns
                if len(episode_paths) == 0 and '**' not in path:
                    # Try adding recursive pattern
                    recursive_path = os.path.join(path, '**', '*.h5')
                    episode_paths.extend(glob.glob(recursive_path, recursive=True))
                    recursive_path = os.path.join(path, '**', '*.hdf5') 
                    episode_paths.extend(glob.glob(recursive_path, recursive=True))
        elif isinstance(path, list):
            episode_paths = path
        else:
            raise ValueError("Path must be a string (file/directory/glob) or a list of strings.")
        
        # Remove duplicates and sort
        episode_paths = sorted(list(set(episode_paths)))
        print(f"Total H5 files to process: {len(episode_paths)}")
        return episode_paths

    def _generate_examples(self, path):
        """Generator of examples for each split."""
        print(path)
        episode_paths = self._discover_h5_files(path)
        
        if len(episode_paths) == 0:
            yield "Empty", {
                'steps': [],
                'episode_metadata': {
                    'file_path': 'No data found at the specified path.'
                }
            }
            return
        
        # Check if running with Beam (multiprocessing)
        if beam is not None:
            try:
                # Return Beam pipeline for parallel processing
                return (
                    beam.Create(episode_paths)
                    | 'ProcessFiles' >> beam.FlatMap(self._process_single_file)
                )
            except Exception:
                # Fall back to sequential processing if Beam fails
                pass
        
        # Sequential processing (fallback or when Beam not available)
        for episode_path in episode_paths:
            yield from self._process_single_file(episode_path)
    
    def _process_single_file(self, episode_path):
        """Process a single HDF5 file and yield all episodes."""
        try:
            with h5py.File(episode_path, 'r') as f:
                all_data = f
                # Extract language instruction from file-level attributes
                language_instruction = "empty language instruction"
                
                # Process each demonstration in the file
                for demo_key in all_data.keys():
                    data = all_data[demo_key]
                    episode = []
                    num_steps = len(data['actions'])
                    
                    for i in range(num_steps):
                        action = data['actions'][i]
                        done = data['terminated'][i] | data['truncated'][i]
                        obs_group = data['obs']
                        left_camera = obs_group['sensor_data']['left_camera']['rgb'][i]
                        right_camera = obs_group['sensor_data']['right_camera']['rgb'][i]
                        hand_camera = obs_group['sensor_data']['hand_camera']['rgb'][i]

                        joint_state = obs_group['agent']['qpos'][i, :7]
                        gripper_state = obs_group['agent']['qpos'][i, 7:]
                        
                        # Convert and process data
                        action = np.asarray(action, dtype=np.float32)
                        done = bool(done)
                        left_camera = np.asarray(left_camera, dtype=np.uint8)
                        right_camera = np.asarray(right_camera, dtype=np.uint8)
                        hand_camera = np.asarray(hand_camera, dtype=np.uint8)
                        joint_state = np.asarray(joint_state, dtype=np.float32)
                        gripper_state = np.asarray(gripper_state, dtype=np.float32)
                        
                        episode.append({
                            'action': action,
                            'is_terminal': done,
                            'is_last': done,
                            'language_instruction': language_instruction,
                            'observation': {
                                'left_camera': left_camera,
                                'right_camera': right_camera,
                                'hand_camera': hand_camera,
                                'joint_state': joint_state,
                                'gripper_state': gripper_state,
                            },
                            'is_first': i == 0,
                            'discount': 1.0,
                            'reward': 1.0 if done else 0.0,
                        })
                    
                    # Create unique ID for each demonstration
                    example_id = f"{episode_path}_{demo_key}"
                    sample = {
                        'steps': episode,
                        'episode_metadata': {
                            'file_path': episode_path
                        }
                    }
                    yield example_id, sample
        except Exception as e:
            print(f"Error processing {episode_path}: {e}")
            return