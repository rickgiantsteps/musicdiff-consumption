import soundfile as sf
import torch
from diffusers import StableAudioPipeline
import pandas as pd
from tqdm import tqdm
import os
from codecarbon import EmissionsTracker
import torch
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

datasets = ["musiccaps_captions.csv", "songdescriber_captions.csv"]

pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16,
                                           token="your_HF_token_here")
pipe = pipe.to(device="cuda", dtype=torch.float16)

negative_prompt = "Low quality."
generator = torch.Generator("cuda").manual_seed(0)

for k in datasets:

    df = pd.read_csv(f"CLAP/{k}")
    filenames = df["file_name"].tolist()
    captions = df["caption"].tolist()
    durations = df["duration"].tolist()

    n_step = [10, 25, 50, 100, 150, 200]

    for i, file in enumerate(tqdm(filenames, desc=f"Processing musicSAO dataset {k}", unit="item")):
        for x in n_step:
                if os.path.exists(f'audios/genaudios/musicSAO/{k.split("_")[0]}/{x}/musicSAO_{x}-steps-'+file):               
                        print(f"{file} already exists, skipping.")
                        continue
                else:

                        tracker = EmissionsTracker(project_name=f"musicSAO_{k.split('_')[0]}-{x}-steps-{file.split('.')[0]}", tracking_mode="process",
                            gpu_ids = "0",
                            output_dir="../../results/quality_metrics/Emissions/musicSAO",
                            output_file=f"musicSAO_{k.split('_')[0]}.csv", allow_multiple_runs=True)
                        tracker.start_task(f"Inference emissions with {x} steps, for {file}")

                        # run the generation
                        audio = pipe(
                                captions[i],
                                negative_prompt=negative_prompt,
                                num_inference_steps=x,
                                audio_end_in_s=10,
                                num_waveforms_per_prompt=1,
                                generator=generator,
                        ).audios

                        model_emissions = tracker.stop_task()
                        _ = tracker.stop()

                        output = audio[0].T.float().cpu().numpy()
                        sf.write(
                        f"audios/genaudios/musicSAO/{k.split('_')[0]}/{x}/musicSAO_{x}-steps-{file}",
                                output, pipe.vae.sampling_rate)

emissions_base = glob.glob('../../results/quality_metrics/Emissions/musicSAO/emissions_base_*')
for f in emissions_base:
    os.remove(f)

print("Done!")
