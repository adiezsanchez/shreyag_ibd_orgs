#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

params.input_dir = file('./images').toAbsolutePath()
params.output_dir = './results'
params.config = "${projectDir}/config.yaml"

process linkOpenCL {
    label 'gpu_test'
    output:
    path "opencl_ready"
    script:
    """
    pixi run fix_opencl
    touch opencl_ready
    """
}

process testGpu {
    tag "gpu_check"
    label 'gpu_test'
    input:
    path opencl_ready
    output:
    path "gpu_ready"
    script:
    """
    pixi run python ${projectDir}/tests/test_gpu.py
    touch gpu_ready
    """
}

process runPipeline {
    tag "${image_file.name}"
    label 'gpu_test'

    publishDir "${params.input_dir}", pattern: "cellpose_labels/*", mode: 'copy'
    publishDir "${projectDir}/results_csv", pattern: "results_csv/*/*.csv", mode: 'copy', saveAs: { filename -> filename.split('/').last() }

    input:
    tuple path(image_file), path(gpu_ready)

    output:
    path "cellpose_labels/*"
    path "results_csv/*/*"

    script:
    """
    pixi run python ${projectDir}/main.py --image "${image_file}" --config ${params.config}
    """
}

workflow {
    images = Channel.fromPath("${params.input_dir}/*.nd2", checkIfExists: true)
    opencl_done = linkOpenCL()
    gpu_done = testGpu(opencl_done)
    run_input = images.combine(gpu_done)
    runPipeline(run_input)
}
