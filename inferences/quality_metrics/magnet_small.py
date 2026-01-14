from audiocraft.models import MAGNeT
from codecarbon import EmissionsTracker
from audiocraft.data.audio import audio_write
import os
import glob
import pandas as pd
from tqdm import tqdm

model = MAGNeT.get_pretrained("facebook/magnet-small-10secs")

datasets = ["musiccaps_captions.csv", "songdescriber_captions.csv"]

for k in datasets:

    df = pd.read_csv(f"CLAP/{k}")
    filenames = df["file_name"].tolist()
    captions = df["caption"].tolist()
    durations = df["duration"].tolist()

    for i, file in enumerate(tqdm(filenames, desc=f"Processing MagnetSmall dataset {k}", unit="item")):

        if os.path.exists(f'audios/genaudios/Magnet/Small/{k.split("_")[0]}/MagnetSmall-'+file):               
                print(f"{file} already exists, skipping.")
                continue
        else:

            descriptions=[captions[i]]

            tracker = EmissionsTracker(project_name=f"MagnetSmall_{k.split('_')[0]}-{file.split('.')[0]}", tracking_mode="process",
                gpu_ids = "0",
                output_dir="../../results/quality_metrics/Emissions/Magnet",
                output_file=f"MagnetSmall_{k.split('_')[0]}.csv", allow_multiple_runs=True)
            tracker.start_task(f"Inference emission, for {file}")

            wav = model.generate(descriptions)

            model_emissions = tracker.stop_task()
            _ = tracker.stop()

            for idx, one_wav in enumerate(wav):
                audio_write(f"audios/genaudios/Magnet/Small/{k.split('_')[0]}/MagnetSmall-{file}",
                             one_wav.cpu(), model.sample_rate, strategy="loudness")

emissions_base = glob.glob('../../results/quality_metrics/Emissions/Magnet/emissions_base_*')
for f in emissions_base:
    os.remove(f)

print("Done!")
