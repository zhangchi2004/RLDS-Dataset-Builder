# RLDS Dataset Conversion

> Chi Zhang <zhang.holomorphic@gmail.com>

This repo demonstrates how to convert an existing dataset into RLDS format for X-embodiment experiment integration.
It is forked from [this repo](https://github.com/kpertsch/rlds_dataset_builder), and specially adapted for our hdf5 formatted rolled-out LIBERO datasets.

## Installation

First create a conda environment using the provided environment.yml file (use `environment_ubuntu.yml` or `environment_macos.yml` depending on the operating system you're using):
```
conda env create -f environment_ubuntu.yml
```

Then activate the environment using:
```
conda activate rlds_env
```

If you want to manually create an environment, the key packages to install are `tensorflow`, 
`tensorflow_datasets`, `tensorflow_hub`, `apache_beam`, `matplotlib`, `plotly` and `wandb`.


## Converting LIBERO from HDF5 to RLDS

To convert your rolled-out LIBERO dataset, follow the steps below:

1. **Rename Dataset**: Change the class name in `example_dataset_dataset_builder.py` (Currently `LiberoBowl`) to the name of your dataset, using camel case instead of underlines.

2. **Select Source Path**: Modify the `SRC_PATH` global variable in `example_dataset_dataset_builder.py` to the path where the .hdf5 files are stored.

3. **Modify Dataset Splits** (if needed in future): The function `_split_generator()` determines the splits of the generated dataset (e.g. training, validation etc.). If you are splitting the train/val sets, add another path for the `'val'` subset.

Inside the dataset directory `example_dataset`, run:
```
tfds build --data_dir=</path/to/your/custom/directory/>
```
This command will generate the RLDS formatted dataset in the path specified in the `--data_dir` argument.


---

# The following content has not been tested.

### Parallelizing Data Processing
By default, dataset conversion is single-threaded. If you are parsing a large dataset, you can use parallel processing.
For this, replace the last two lines of `_generate_examples()` with the commented-out `beam` commands. This will use 
Apache Beam to parallelize data processing. Before starting the processing, you need to install your dataset package 
by filling in the name of your dataset into `setup.py` and running `pip install -e .`

Then, make sure that no GPUs are used during data processing (`export CUDA_VISIBLE_DEVICES=`) and run:
```
tfds build --overwrite --beam_pipeline_options="direct_running_mode=multi_processing,direct_num_workers=10"
```
You can specify the desired number of workers with the `direct_num_workers` argument.

## Visualize Converted Dataset
To verify that the data is converted correctly, please run the data visualization script from the base directory:
```
python3 visualize_dataset.py <name_of_your_dataset>
``` 
This will display a few random episodes from the dataset with language commands and visualize action and state histograms per dimension.
Note, if you are running on a headless server you can modify `WANDB_ENTITY` at the top of `visualize_dataset.py` and 
add your own WandB entity -- then the script will log all visualizations to WandB. 

## Add Transform for Target Spec

For X-embodiment training we are using specific inputs / outputs for the model: input is a single RGB camera, output
is an 8-dimensional action, consisting of end-effector position and orientation, gripper open/close and a episode termination
action.

The final step in adding your dataset to the training mix is to provide a transform function, that transforms a step
from your original dataset above to the required training spec. Please follow the two simple steps below:

1. **Modify Step Transform**: Modify the function `transform_step()` in `example_transform/transform.py`. The function 
takes in a step from your dataset above and is supposed to map it to the desired output spec. The file contains a detailed
description of the desired output spec.

2. **Test Transform**: We provide a script to verify that the resulting __transformed__ dataset outputs match the desired
output spec. Please run the following command: `python3 test_dataset_transform.py <name_of_your_dataset>`

If the test passes successfully, you are ready to upload your dataset!

## Upload Your Data

We provide a Google Cloud bucket that you can upload your data to. First, install `gsutil`, the Google cloud command 
line tool. You can follow the installation instructions [here](https://cloud.google.com/storage/docs/gsutil_install).

Next, authenticate your Google account with:
```
gcloud auth login
``` 
This will open a browser window that allows you to log into your Google account (if you're on a headless server, 
you can add the `--no-launch-browser` flag). Ideally, use the email address that
you used to communicate with Karl, since he will automatically grant permission to the bucket for this email address. 
If you want to upload data with a different email address / google account, please shoot Karl a quick email to ask 
to grant permissions to that Google account!

After logging in with a Google account that has access permissions, you can upload your data with the following 
command:
```
gsutil -m cp -r ~/tensorflow_datasets/<name_of_your_dataset> gs://xembodiment_data
``` 
This will upload all data using multiple threads. If your internet connection gets interrupted anytime during the upload
you can just rerun the command and it will resume the upload where it was interrupted. You can verify that the upload
was successful by inspecting the bucket [here](https://console.cloud.google.com/storage/browser/xembodiment_data).

The last step is to commit all changes to this repo and send Karl the link to the repo.

**Thanks a lot for contributing your data! :)**
