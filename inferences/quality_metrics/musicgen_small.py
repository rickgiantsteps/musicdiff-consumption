from transformers import AutoProcessor, MusicgenForConditionalGeneration
from codecarbon import EmissionsTracker
import scipy
import os
import glob
import glob
import pandas as pd
from tqdm import tqdm

processor = AutoProcessor.from_pretrained("facebook/musicgen-small", device='cuda')
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
model = model.to('cuda:0')

datasets = ["musiccaps_captions.csv", "songdescriber_captions.csv"]

for k in datasets:

    df = pd.read_csv(f"CLAP/{k}")
    filenames = df["file_name"].tolist()
    captions = df["caption"].tolist()
    durations = df["duration"].tolist()

    for i, file in enumerate(tqdm(filenames, desc=f"Processing MusicGenSmall dataset {k}", unit="item")):

        if os.path.exists(f'audios/genaudios/MusicGen/Small/{k.split("_")[0]}/MusicGenSmall-'+file):               
                print(f"{file} already exists, skipping.")
                continue
        else:

            inputs = processor(
                text=[captions[i]],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to('cuda:0')

            tracker = EmissionsTracker(project_name=f"MusicGenSmall_{k.split('_')[0]}-{file.split('.')[0]}", tracking_mode="process",
                gpu_ids = "0",
                output_dir="../../results/quality_metrics/Emissions/MusicGen",
                output_file=f"MusicGenSmall_{k.split('_')[0]}.csv", allow_multiple_runs=True)
            tracker.start_task(f"Inference emission, for {file}")

            audio_values = model.generate(**inputs, max_new_tokens=500)

            model_emissions = tracker.stop_task()
            _ = tracker.stop()

            sampling_rate = model.config.audio_encoder.sampling_rate
            scipy.io.wavfile.write(f"audios/genaudios/MusicGen/Small/{k.split('_')[0]}/MusicGenSmall-{file}",
                                rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())

emissions_base = glob.glob('../../results/quality_metrics/Emissions/MusicGen/emissions_base_*')
for f in emissions_base:
    os.remove(f)

print("Done!")
