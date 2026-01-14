import click
import os
import glob
import contextlib
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


    n_step = [10, 25, 50, 100, 150, 200]
    runs = 5

    warmup_step = n_step[0]
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
        for x in n_step:

            tracker = EmissionsTracker(project_name=f"ACEStep-inference_{x}-steps", tracking_mode="process",
                                    gpu_ids="0",
                                    output_dir="../../results/inference_steps/ACEStep",
                                    output_file=f"ACEStep-emissions-run{k+1}.csv", allow_multiple_runs=True)
            tracker.start_task(f"Inference emissions with {x} steps")

            infer_step = x   

            model_demo(
                batch_size = batch_size,
                audio_duration=audio_duration,
                prompt=prompt,
                lyrics=lyrics,
                infer_step=infer_step,
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
                save_path=f"genaudios/ACEStep_{x}-steps.wav",
            )

            model_emissions = tracker.stop_task()
            _ = tracker.stop()

            print(model_emissions)

    emissions_base = glob.glob('../../results/inference_steps/ACEStep/emissions_base_*')
    for f in emissions_base:
        os.remove(f)

    json_files = glob.glob("genaudios/*.json")
    for f in json_files:
        os.remove(f)

    print("Done!")


if __name__ == "__main__":
    main()
