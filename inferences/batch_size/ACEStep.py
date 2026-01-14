import click
import os
import glob
import contextlib
from itertools import chain
from codecarbon import EmissionsTracker
from acestep.pipeline_ace_step import ACEStepPipeline


@click.command()
@click.option(
    "--checkpoint_path", type=str, default="", help="Path to the checkpoint directory"
)
@click.option("--bf16", type=bool, default=True, help="Whether to use bfloat16")
@click.option(
    "--torch_compile", type=bool, default=False, help="Whether to use torch compile"
)
@click.option(
    "--cpu_offload", type=bool, default=False, help="Whether to use CPU offloading (only load current stage's model to GPU)"
)
@click.option(
    "--overlapped_decode", type=bool, default=False, help="Whether to use overlapped decoding (run dcae and vocoder using sliding windows)"
)


def main(checkpoint_path, bf16, torch_compile, cpu_offload, overlapped_decode):

    emissions = glob.glob('../../results/batch_size/ACEStep/*')
    emissions1 = glob.glob('../../results/batch_size/ACEStep/batch1/*')
    emissions2 = glob.glob('../../results/batch_size/ACEStep/batch2/*')
    emissions5 = glob.glob('../../results/batch_size/ACEStep/batch5/*')
    genaudios = glob.glob('genaudios/ACEStep/*')
    for f in chain(emissions, emissions1, emissions2, emissions5, genaudios):
        if os.path.isfile(f):
            os.remove(f)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model_demo = ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype="bfloat16" if bf16 else "float32",
        torch_compile=torch_compile,
        cpu_offload=cpu_offload,
        overlapped_decode=overlapped_decode
    )
    print(model_demo)

    audio_duration = 10.0
    prompt = "An organ is playing chords and melodies while a male voice is singing soulfully."
    lyrics = ""
    guidance_scale = 15
    #scheduler_type = "ddim"
    scheduler_type = "euler"
    cfg_type = "apg"
    omega_scale = 10
    manual_seeds = "1318394052"
    guidance_interval = 0.5
    guidance_interval_decay = 0
    min_guidance_scale = 3
    use_erg_tag = False
    use_erg_lyric = False
    use_erg_diffusion = False
    oss_steps = ''
    guidance_scale_text = 0.0
    guidance_scale_lyric = 0.0
    batch_size = 1

    runs = 5

    warmup_step = 10
    try:
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                model_demo(
                    audio_duration=audio_duration,
                    prompt=prompt,
                    lyrics=lyrics,
                    infer_step=warmup_step,
                    guidance_scale=guidance_scale,
                    scheduler_type=scheduler_type,
                    cfg_type=cfg_type,
                    omega_scale=omega_scale,
                    manual_seeds=manual_seeds,
                    guidance_interval=guidance_interval,
                    guidance_interval_decay=guidance_interval_decay,
                    min_guidance_scale=min_guidance_scale,
                    use_erg_tag=use_erg_tag,
                    use_erg_lyric=use_erg_lyric,
                    use_erg_diffusion=use_erg_diffusion,
                    oss_steps=oss_steps,
                    guidance_scale_text=guidance_scale_text,
                    guidance_scale_lyric=guidance_scale_lyric,
                    save_path=None,  # no saving for warm-up
                )
    except Exception as e:
        # If warmup fails, print the error so you can debug
        print("Warm-up run failed:", e)
        raise

    for k in range(runs):

        if k==0:
            outpath = "genaudios/ACEStep"
        else:
            outpath = None

        tracker = EmissionsTracker(project_name=f"ACEStep-inference_batch10", tracking_mode="process",
                                gpu_ids="0",
                                output_dir="../../results/batch_size/ACEStep",
                                output_file=f"ACEStep-emissions-batch10-run{k+1}.csv", allow_multiple_runs=True)
        tracker.start_task(f"Inference emissions with 10 waveforms per prompt")  

        model_demo(
            batch_size = 10,
            audio_duration=audio_duration,
            prompt=prompt,
            lyrics=lyrics,
            infer_step=100,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            omega_scale=omega_scale,
            manual_seeds=manual_seeds,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            min_guidance_scale=min_guidance_scale,
            use_erg_tag=use_erg_tag,
            use_erg_lyric=use_erg_lyric,
            use_erg_diffusion=use_erg_diffusion,
            oss_steps=oss_steps,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            save_path=outpath,
        )

        model_emissions = tracker.stop_task()
        _ = tracker.stop()

        print(model_emissions)

    for k in range(runs):
        for i in range(2):

            if k==0:
                outpath = "genaudios/ACEStep"
            else:
                outpath = None

            tracker = EmissionsTracker(project_name=f"ACEStep-inference_batch5", tracking_mode="process",
                                    gpu_ids="0",
                                    output_dir="../../results/batch_size/ACEStep/batch5",
                                    output_file=f"ACEStep-emissions-batch5-n{i}-run{k+1}.csv", allow_multiple_runs=True)
            tracker.start_task(f"Inference emissions with 5 waveforms per prompt")  

            model_demo(
                batch_size = 5,
                audio_duration=audio_duration,
                prompt=prompt,
                lyrics=lyrics,
                infer_step=100,
                guidance_scale=guidance_scale,
                scheduler_type=scheduler_type,
                cfg_type=cfg_type,
                omega_scale=omega_scale,
                manual_seeds=manual_seeds,
                guidance_interval=guidance_interval,
                guidance_interval_decay=guidance_interval_decay,
                min_guidance_scale=min_guidance_scale,
                use_erg_tag=use_erg_tag,
                use_erg_lyric=use_erg_lyric,
                use_erg_diffusion=use_erg_diffusion,
                oss_steps=oss_steps,
                guidance_scale_text=guidance_scale_text,
                guidance_scale_lyric=guidance_scale_lyric,
                save_path=outpath,
            )

            model_emissions = tracker.stop_task()
            _ = tracker.stop()

        print(model_emissions)

    for k in range(runs):
        for i in range(5):

            if k==0:
                outpath = "genaudios/ACEStep"
            else:
                outpath = None

            tracker = EmissionsTracker(project_name=f"ACEStep-inference_batch2", tracking_mode="process",
                                    gpu_ids="0",
                                    output_dir="../../results/batch_size/ACEStep/batch2",
                                    output_file=f"ACEStep-emissions-batch2-n{i}-run{k+1}.csv", allow_multiple_runs=True)
            tracker.start_task(f"Inference emissions with 2 waveforms per prompt")  

            model_demo(
                batch_size = 2,
                audio_duration=audio_duration,
                prompt=prompt,
                lyrics=lyrics,
                infer_step=100,
                guidance_scale=guidance_scale,
                scheduler_type=scheduler_type,
                cfg_type=cfg_type,
                omega_scale=omega_scale,
                manual_seeds=manual_seeds,
                guidance_interval=guidance_interval,
                guidance_interval_decay=guidance_interval_decay,
                min_guidance_scale=min_guidance_scale,
                use_erg_tag=use_erg_tag,
                use_erg_lyric=use_erg_lyric,
                use_erg_diffusion=use_erg_diffusion,
                oss_steps=oss_steps,
                guidance_scale_text=guidance_scale_text,
                guidance_scale_lyric=guidance_scale_lyric,
                save_path=outpath,
            )

            model_emissions = tracker.stop_task()
            _ = tracker.stop()

        print(model_emissions)

    for k in range(runs):
        for i in range(10):

            if k==0:
                outpath = "genaudios/ACEStep"
            else:
                outpath = None

            tracker = EmissionsTracker(project_name=f"ACEStep-inference_batch1", tracking_mode="process",
                                    gpu_ids="0",
                                    output_dir="../../results/batch_size/ACEStep/batch1",
                                    output_file=f"ACEStep-emissions-batch1-n{i}-run{k+1}.csv", allow_multiple_runs=True)
            tracker.start_task(f"Inference emissions with 1 waveforms per prompt")  

            model_demo(
                batch_size = 1,
                audio_duration=audio_duration,
                prompt=prompt,
                lyrics=lyrics,
                infer_step=100,
                guidance_scale=guidance_scale,
                scheduler_type=scheduler_type,
                cfg_type=cfg_type,
                omega_scale=omega_scale,
                manual_seeds=manual_seeds,
                guidance_interval=guidance_interval,
                guidance_interval_decay=guidance_interval_decay,
                min_guidance_scale=min_guidance_scale,
                use_erg_tag=use_erg_tag,
                use_erg_lyric=use_erg_lyric,
                use_erg_diffusion=use_erg_diffusion,
                oss_steps=oss_steps,
                guidance_scale_text=guidance_scale_text,
                guidance_scale_lyric=guidance_scale_lyric,
                save_path=outpath,
            )

            model_emissions = tracker.stop_task()
            _ = tracker.stop()

        print(model_emissions)


    emissions_base = glob.glob('../../results/batch_size/ACEStep/emissions_base_*')
    for f in emissions_base:
        os.remove(f)

    emissions_base = glob.glob('../../results/batch_size/ACEStep/batch5/emissions_base_*')
    for f in emissions_base:
        os.remove(f)

    emissions_base = glob.glob('../../results/batch_size/ACEStep/batch2/emissions_base_*')
    for f in emissions_base:
        os.remove(f)

    emissions_base = glob.glob('../../results/batch_size/ACEStep/batch1/emissions_base_*')
    for f in emissions_base:
        os.remove(f)

    json_files = glob.glob("genaudios/ACEStep/*.json")
    for f in json_files:
        os.remove(f)

    unnecessaryout = glob.glob("genaudios/ACEStep/outputs/*")
    for f in unnecessaryout:
        os.remove(f)

    print("Done!")


if __name__ == "__main__":
    main()
