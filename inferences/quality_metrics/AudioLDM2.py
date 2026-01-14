import scipy
import torch
from diffusers import AudioLDM2Pipeline
import pandas as pd
from tqdm import tqdm
import os
from codecarbon import EmissionsTracker
import torch
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

datasets = ["musiccaps_captions.csv", "songdescriber_captions.csv"]

repo_id = "cvssp/audioldm2-music"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to(device="cuda", dtype=torch.float16)
negative_prompt = "Low quality."
generator = torch.Generator("cuda").manual_seed(0)

for k in datasets:

    df = pd.read_csv(f"CLAP/{k}")
    filenames = df["file_name"].tolist()
    captions = df["caption"].tolist()
    durations = df["duration"].tolist()

    n_step = [10, 25, 50, 100, 150, 200]

    for i, file in enumerate(tqdm(filenames, desc=f"Processing musicAudioLDM2 dataset {k}", unit="item")):
        for x in n_step:
            if os.path.exists(f'audios/genaudios/musicAudioLDM2/{k.split("_")[0]}/{x}/musicAudioLDM2_{x}-steps-'+file):                       
                print(f"{file} already exists, skipping.")
                continue
            else:

                tracker = EmissionsTracker(project_name=f"musicAudioLDM2_{k.split('_')[0]}-{x}-steps-{file.split('.')[0]}", tracking_mode="process",
                    gpu_ids = "0",
                    output_dir="../../results/quality_metrics/Emissions/musicAudioLDM2",
                    output_file=f"musicAudioLDM2_{k.split('_')[0]}.csv", allow_multiple_runs=True)
                tracker.start_task(f"Inference emissions with {x} steps, for {file}")

                # run the generation
                audio = pipe(
                        captions[i],
                        negative_prompt=negative_prompt,
                        num_inference_steps=x,
                        audio_length_in_s=10,
                        num_waveforms_per_prompt=1,
                        generator=generator,
                ).audios

                model_emissions = tracker.stop_task()
                _ = tracker.stop()

                scipy.io.wavfile.write(
                    f"audios/genaudios/musicAudioLDM2/{k.split('_')[0]}/{x}/musicAudioLDM2_{x}-steps-{file}",
                    rate=16000, data=audio[0])
                print(f"{x} steps, {captions[i]}")

emissions_base = glob.glob('../../results/quality_metrics/Emissions/musicAudioLDM2/emissions_base_*')
for f in emissions_base:
    os.remove(f)

print("Done!")
