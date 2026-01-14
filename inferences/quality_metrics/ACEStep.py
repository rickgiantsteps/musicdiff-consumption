import click
import os
import glob
import contextlib
import pandas as pd
from tqdm import tqdm
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
        print("Warm-up run failed:", e)
        raise

    datasets = ["musiccaps_captions.csv", "songdescriber_captions.csv"]

    for k in datasets:

        df = pd.read_csv(f"CLAP/{k}")
        filenames = df["file_name"].tolist()
        captions = df["caption"].tolist()
        durations = df["duration"].tolist()

        for i, file in enumerate(tqdm(filenames, desc=f"Processing ACEStep dataset {k}", unit="item")):
            for x in n_step:
                    if os.path.exists(f'audios/genaudios/ACEStep/{k.split("_")[0]}/{x}/ACEStep_{x}-steps-'+file):               
                            print(f"{file} already exists, skipping.")
                            continue
                    else:

                        tracker = EmissionsTracker(project_name=f"ACEStep_{k.split('_')[0]}-{x}-steps-{file.split('.')[0]}", tracking_mode="process",
                            gpu_ids = "0",
                            output_dir="../../results/quality_metrics/Emissions/ACEStep",
                            output_file=f"ACEStep_{k.split('_')[0]}.csv", allow_multiple_runs=True)
                        tracker.start_task(f"Inference emissions with {x} steps, for {file}")


                        model_demo(
                            batch_size = batch_size,
                            audio_duration=audio_duration,
                            prompt=captions[i],
                            lyrics=lyrics,
                            infer_step=x,
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
                            save_path=f"audios/genaudios/ACEStep/{k.split('_')[0]}/{x}/ACEStep_{x}-steps-{file}",
                        )

                        model_emissions = tracker.stop_task()
                        _ = tracker.stop()

                        print(model_emissions)

    emissions_base = glob.glob('../../results/quality_metrics/Emissions/ACEStep/emissions_base_*')
    for f in emissions_base:
        os.remove(f)

    for x in n_step:
        json_files = glob.glob(f"audios/genaudios/ACEStep/musiccaps/{x}/*.json")
        for f in json_files:
            os.remove(f)
        json_files = glob.glob(f"audios/genaudios/ACEStep/songdescriber/{x}/*.json")
        for f in json_files:
            os.remove(f)

    print("Done!")


if __name__ == "__main__":
    main()
