import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.data.utils import read_video
from stable_audio_tools.data.utils import load_and_process_audio
from codecarbon import EmissionsTracker
import glob
import os

device = torch.device("cuda:0")

datasets = ["musiccaps_captions.csv", "songdescriber_captions.csv"]

# Download model
model, model_config = get_pretrained_model("HKUSTAudio/AudioX")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
target_fps = model_config["video_fps"]
seconds_start = 0
seconds_total = 10

model = model.to(device)

# only text-to-music generation
video_path = None
audio_path = None

video_tensor = read_video(video_path, seek_time=0, duration=seconds_total, target_fps=target_fps)
audio_tensor = load_and_process_audio(audio_path, sample_rate, seconds_start, seconds_total)
    
for k in datasets:

    df = pd.read_csv(f"CLAP/{k}")
    filenames = df["file_name"].tolist()
    captions = df["caption"].tolist()
    durations = df["duration"].tolist()

    n_step = [10, 25, 50, 100, 150, 200]

    for i, file in enumerate(tqdm(filenames, desc=f"Processing AudioX dataset {k}", unit="item")):
        for x in n_step:
            if os.path.exists(f'audios/genaudios/AudioX/{k.split("_")[0]}/{x}/AudioX_{x}-steps-'+file):               
                    print(f"{file} already exists, skipping.")
                    continue
            else:

                conditioning = [{
                    "video_prompt": [video_tensor.unsqueeze(0)],        
                    "text_prompt": captions[i],
                    "audio_prompt": audio_tensor.unsqueeze(0),
                    "seconds_start": seconds_start,
                    "seconds_total": seconds_total
                }]

                tracker = EmissionsTracker(project_name=f"AudioX_{k.split('_')[0]}-{x}-steps-{file.split('.')[0]}", tracking_mode="process",
                    gpu_ids = "0",
                    output_dir="../../results/quality_metrics/Emissions/AudioX",
                    output_file=f"AudioX_{k.split('_')[0]}.csv", allow_multiple_runs=True)
                tracker.start_task(f"Inference emissions with {x} steps, for {file}")

                # Generate stereo audio
                output = generate_diffusion_cond(
                    model,
                    steps=x,
                    cfg_scale=7,
                    conditioning=conditioning,
                    sample_size=sample_size,
                    sigma_min=0.3,
                    sigma_max=500,
                    sampler_type="dpmpp-3m-sde",
                    device=device
                )

                model_emissions = tracker.stop_task()
                _ = tracker.stop()

                # Rearrange audio batch to a single sequence
                output = rearrange(output, "b d n -> d (b n)")

                # Peak normalize, clip, convert to int16, and save to file
                output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                torchaudio.save(f"audios/genaudios/AudioX/{k.split('_')[0]}/{x}/AudioX_{x}-steps-{file}", output, sample_rate)
                print(model_emissions)

emissions_base = glob.glob('../../results/quality_metrics/Emissions/AudioX/emissions_base_*')
for f in emissions_base:
    os.remove(f)

print("Done!")
