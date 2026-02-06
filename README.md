# shreyag_ibd_orgs
Analysis of colon organoids in 2D and 3D 

After cloning the repo, in your terminal run:

This analyzes a single image and is meant to be run using Nextflow for parallelization:
Input variables for analysis (i.e. which cell compartments and channels to analyze need to be edited in the config.yaml file), alternatively via arguments (--)

<code>cd shreya_ibd_orgs</code>

<code>pixi shell</code>
<code>python main.py --image "Z:\lusie.f.kuraas\PhD\Nikon Spinning Disc\20260114_T7_2microns\B2.nd2" --config config.yaml</code>

Or directly:
<code>pixi run python main.py --image "Z:\lusie.f.kuraas\PhD\Nikon Spinning Disc\20260114_T7_2microns\B2.nd2" --config config.yaml</code>

There are also interactive Jupyter Notebooks for batch processing of images using for loops under <code>notebooks</code>

#TODO: Adapt pixi environment to run on Linux
#TODO: Create Nextflow config files to run in HPC using slurm (do not hardcode so it can work on Azure or AWS)