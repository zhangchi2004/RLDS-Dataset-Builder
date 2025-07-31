from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

import h5py

# By zc
SRC_PATH = '/home/projects/xlang.slurm/' # Modify this path to your dataset location

class LiberoBasket(tfds.core.GeneratorBasedBuilder): # Modify the class name to your dataset name
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
                        'wrist_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB observation.',
                        ),
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc='Robot EEF state (6D pose, 2D gripper).',
                        ),
                        'joint_state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint angles.',
                        ),
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
            'train': self._generate_examples(path=f'{SRC_PATH}/*.hdf5'),
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

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        if isinstance(path, str):
            episode_paths = glob.glob(path)
        elif isinstance(path, list):
            episode_paths = path
        else:
            raise ValueError("Path must be a string or a list of strings.")
        if len(episode_paths) == 0:
            yield "Empty", {
                'steps': [],
                'episode_metadata': {
                    'file_path': 'No data found at the specified path.'
                }
            }
        for episode_path in episode_paths:
            with h5py.File(episode_path, 'r') as f:
                all_data = f['data']
                # Extract language instruction from file-level attributes
                lang_info = all_data.attrs['problem_info']
                language_instruction = lang_info.split('"language_instruction": "')[1].split('"')[0]
                
                # Process each demonstration in the file
                for demo_key in all_data.keys():
                    data = all_data[demo_key]
                    episode = []
                    num_steps = len(data['actions'])
                    
                    for i in range(num_steps):
                        action = data['actions'][i]
                        done = data['dones'][i]
                        obs_group = data['obs']
                        agentview_rgb = obs_group['agentview_rgb'][i]
                        ee_states = obs_group['ee_states'][i]
                        eye_in_hand_rgb = obs_group['eye_in_hand_rgb'][i]
                        gripper_state = obs_group['gripper_states'][i]
                        joint_state = obs_group['joint_states'][i]
                        
                        # Convert and process data
                        action = np.asarray(action, dtype=np.float32)
                        done = bool(done)
                        image = np.asarray(agentview_rgb, dtype=np.uint8)[::-1, ::-1]
                        wrist_image = np.asarray(eye_in_hand_rgb, dtype=np.uint8)[::-1, ::-1]
                        ee_states = np.asarray(ee_states, dtype=np.float32)
                        gripper_state = np.asarray(gripper_state, dtype=np.float32)
                        state = np.concatenate((ee_states, gripper_state), axis=0)
                        joint_state = np.asarray(joint_state, dtype=np.float32)
                        
                        episode.append({
                            'action': action,
                            'is_terminal': done,
                            'is_last': done,
                            'language_instruction': language_instruction,
                            'observation': {
                                'wrist_image': wrist_image,
                                'image': image,
                                'state': state,
                                'joint_state': joint_state,
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