import scipy
import torch
import os
import glob
from itertools import chain
from diffusers import MusicLDMPipeline
from codecarbon import EmissionsTracker

emissions = glob.glob('../../results/batch_size/MusicLDM/*')
emissions1 = glob.glob('../../results/batch_size/MusicLDM/batch1/*')
emissions2 = glob.glob('../../results/batch_size/MusicLDM/batch2/*')
emissions5 = glob.glob('../../results/batch_size/MusicLDM/batch5/*')
genaudios = glob.glob('genaudios/MusicLDM/*')
for f in chain(emissions, emissions1, emissions2, emissions5, genaudios):
    if os.path.isfile(f):
        os.remove(f)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

repo_id = "ucsd-reach/musicldm"
pipe = MusicLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to(device="cuda", dtype=torch.float16)

prompt = "An organ is playing chords and melodies while a male voice is singing soulfully."
negative_prompt = "Low quality."
generator = torch.Generator("cuda").manual_seed(0)

runs = 5

for k in range(runs):

    tracker = EmissionsTracker(project_name=f"MusicLDM-inference_batch10", tracking_mode="process",
                               gpu_ids="0",
                               output_dir="../../results/batch_size/MusicLDM",
                               output_file=f"MusicLDM-emissions-batch10-run{k+1}.csv", allow_multiple_runs=True)
    tracker.start_task(f"Inference emissions with 10 waveforms per prompt")

    try:

        audio = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=100,
            audio_length_in_s=10.0,
            num_waveforms_per_prompt=10,
            generator=generator,
        ).audios

        model_emissions = tracker.stop_task()

    finally:
        _ = tracker.stop()

    for i in range(10):
        scipy.io.wavfile.write(
            f"genaudios/MusicLDM/MusicLDM-batch10-n{i}.wav",
            rate=16000, data=audio[i])
        print(model_emissions)

for k in range(runs):
    file_counter = 0
    for i in range(2):
        tracker = EmissionsTracker(project_name=f"MusicLDM-inference_batch5", tracking_mode="process",
                                   gpu_ids="0",
                                   output_dir="../../results/batch_size/MusicLDM/batch5",
                                   output_file=f"MusicLDM-emissions-batch5-n{i}-run{k+1}.csv", allow_multiple_runs=True)
        tracker.start_task(f"Inference emissions with 1 waveform per prompt")

        try:

            audio = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=100,
                audio_length_in_s=10.0,
                num_waveforms_per_prompt=5,
                generator=generator,
            ).audios

            model_emissions = tracker.stop_task()

        finally:
            _ = tracker.stop()

        for x in range(5):
            scipy.io.wavfile.write(
                f"genaudios/MusicLDM/MusicLDM-batch5-n{file_counter}.wav",
                rate=16000, data=audio[x])
            file_counter += 1
        print(model_emissions)


for k in range(runs):
    file_counter = 0
    for i in range(5):
        tracker = EmissionsTracker(project_name=f"MusicLDM-inference_batch2", tracking_mode="process",
                                   gpu_ids="0",
                                   output_dir="../../results/batch_size/MusicLDM/batch2",
                                   output_file=f"MusicLDM-emissions-batch2-n{i}-run{k+1}.csv", allow_multiple_runs=True)
        tracker.start_task(f"Inference emissions with 1 waveform per prompt")

        try:

            audio = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=100,
                audio_length_in_s=10.0,
                generator=generator,
                num_waveforms_per_prompt=2,
            ).audios

            model_emissions = tracker.stop_task()

        finally:
            _ = tracker.stop()

        for x in range(2):
            scipy.io.wavfile.write(
            f"genaudios/MusicLDM/MusicLDM-batch2-n{file_counter}.wav",
            rate=16000, data=audio[x])
            file_counter += 1
        print(model_emissions)

for k in range(runs):
    for i in range(10):
        tracker = EmissionsTracker(project_name=f"MusicLDM-inference_batch1", tracking_mode="process",
                                   gpu_ids="0",
                                   output_dir="../../results/batch_size/MusicLDM/batch1",
                                   output_file=f"MusicLDM-emissions-batch1-n{i}-run{k+1}.csv", allow_multiple_runs=True)
        tracker.start_task(f"Inference emissions with 1 waveform per prompt")

        try:

            audio = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=100,
                audio_length_in_s=10.0,
                num_waveforms_per_prompt=1,
                generator=generator,
            ).audios

            model_emissions = tracker.stop_task()

        finally:
            _ = tracker.stop()

        scipy.io.wavfile.write(
            f"genaudios/MusicLDM/MusicLDM-batch1-n{i}.wav",
            rate=16000, data=audio[0])
        print(model_emissions)


emissions_base = glob.glob('../../results/batch_size/MusicLDM/emissions_base_*')
for f in emissions_base:
    os.remove(f)

emissions_base = glob.glob('../../results/batch_size/MusicLDM/batch5/emissions_base_*')
for f in emissions_base:
    os.remove(f)

emissions_base = glob.glob('../../results/batch_size/MusicLDM/batch2/emissions_base_*')
for f in emissions_base:
    os.remove(f)

emissions_base = glob.glob('../../results/batch_size/MusicLDM/batch1/emissions_base_*')
for f in emissions_base:
    os.remove(f)

print("Done!")
