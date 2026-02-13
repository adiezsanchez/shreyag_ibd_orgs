#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

params.input_dir = file('./images').toAbsolutePath()
params.output_dir = './results'
params.config = "${projectDir}/config.yaml"

process runPipeline {
    tag "${image_file.name}"
    label 'gpu_test'

    publishDir "${params.input_dir}", pattern: "cellpose_labels/*", mode: 'copy'
    publishDir "${projectDir}/results_csv", pattern: "results_csv/*/*.csv", mode: 'copy', saveAs: { filename -> filename.split('/').last() }

    input:
    path image_file

    output:
    path "cellpose_labels/*"
    path "results_csv/*/*"

    script:
    """
    module load CUDA/12.1.1 2>/dev/null || true
    export OPENCL_VENDOR_PATH="${projectDir}/.pixi/envs/default/etc/OpenCL/vendors:/etc/OpenCL/vendors"
    pixi run fix_opencl_hpc
    pixi run python ${projectDir}/main.py --image "${image_file}" --config ${params.config}
    """
}

workflow {
    images = Channel.fromPath("${params.input_dir}/*.nd2", checkIfExists: true)
    runPipeline(images)
}
