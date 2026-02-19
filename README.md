# shreyag_ibd_orgs

Analysis of colon organoids in 2D and 3D: Cellpose-based segmentation, organoid detection, and per-cell/per-organoid measurements from Nikon ND2 and Zeiss CZI images. Inputs (markers, channels, compartments) are set in `config.yaml` or via CLI.

**Requirements:** [pixi](https://pixi.sh/), Git and a CUDA-capable GPU (for Cellpose and OpenCL).

---

## Running locally

From the project root, install the environment and run the pipeline.

If you are not familiar with command line the easiest will be to use Jupyter Notebooks, just copy and paste the following command in your terminal (i.e. cmd for Windows):

```bash
git clone https://github.com/adiezsanchez/shreyag_ibd_orgs.git && cd shreya_ibd_orgs && pixi run lab
```

This will clone the repo, install the virtual environment via pixi and launch a Jupyter instance to interact with the Jupyter Notebooks inside ./notebooks.

### Single image

```bash
pixi run python main.py --image "/path/to/image.nd2" --config config.yaml
```

Or after entering the environment:

```bash
pixi shell
python main.py --image "/path/to/image.nd2" --config config.yaml
```

### #TODO: Whole folder (all `.nd2` in a directory)

Use pixi tasks so the environment is used automatically.

**Windows (PowerShell):**

```powershell
pixi run run_folder_powershell "C:\path\to\image\folder"
pixi run run_folder_powershell "C:\path\to\image\folder" other_config.yaml
```

**Linux / macOS / Git Bash / WSL (Bash):**

```bash
pixi run run_folder_bash /path/to/image/folder
pixi run run_folder_bash /path/to/image/folder other_config.yaml
```

**Cross-platform (pixi picks the script by OS):**

```bash
pixi run run_folder /path/to/image/folder
pixi run run_folder /path/to/image/folder other_config.yaml
```

Arguments: first = image folder path (required), second = config file (optional, default `config.yaml`).

---

## Jupyter notebooks

Interactive batch runs and exploration:

```bash
pixi run lab
```

Then open the notebooks under `notebooks/` (e.g. `BP_organoid_analysis.ipynb`, `SP_organoid_analysis.ipynb`). They loop over images in a folder and run the same pipeline logic interactively.

---

## IDUN HPC (NTNU)

On IDUN, use Slurm to process a directory of images in parallel on GPUs. From the project directory on the cluster:

```bash
# Process all .nd2 in a directory (default config: config.yaml)
sbatch hpc_sbatch/run_pipeline_slurm.sh /path/to/directory_with_nd2

# Optional: specify config path
sbatch hpc_sbatch/run_pipeline_slurm.sh /path/to/directory_with_nd2 config.yaml
```

Example with project data path:

```bash
sbatch hpc_sbatch/run_pipeline_slurm.sh /cluster/work/albertds/raw_data/20260114_T7_2microns
```

Check job status:

```bash
squeue -u $USER
```

Edit `hpc_sbatch/run_pipeline_slurm.sh` to set `PROJECT_DIR` and Slurm account/partition if needed. Other scripts in `hpc_sbatch/` target different GPU types (A100, H100, P100, V100) just for testing OpenCL and CUDA acceleration.
