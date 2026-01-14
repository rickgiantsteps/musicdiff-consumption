import scipy
import torch
import os
import glob
from itertools import chain
from diffusers import AudioLDMPipeline
from codecarbon import EmissionsTracker

emissions = glob.glob('../../results/batch_size/musicAudioLDM/*')
emissions1 = glob.glob('../../results/batch_size/musicAudioLDM/batch1/*')
emissions2 = glob.glob('../../results/batch_size/musicAudioLDM/batch2/*')
emissions5 = glob.glob('../../results/batch_size/musicAudioLDM/batch5/*')
genaudios = glob.glob('genaudios/musicAudioLDM/*')
for f in chain(emissions, emissions1, emissions5, genaudios):
    if os.path.isfile(f):
        os.remove(f)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

repo_id = "cvssp/audioldm-s-full-v2"
pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to(device="cuda", dtype=torch.float16)

prompt = "An organ is playing chords and melodies while a male voice is singing soulfully."
negative_prompt = "Low quality."
generator = torch.Generator("cuda").manual_seed(0)

runs = 5

for k in range(runs):

    tracker = EmissionsTracker(project_name=f"musicAudioLDM-inference_batch10", tracking_mode="process",
                               gpu_ids="0",
                               output_dir="../../results/batch_size/musicAudioLDM",
                               output_file=f"musicAudioLDM-emissions-batch10-run{k+1}.csv", allow_multiple_runs=True)
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
            f"genaudios/musicAudioLDM/musicAudioLDM-batch10-n{i}.wav",
            rate=16000, data=audio[i])
        print(model_emissions)

for k in range(runs):
    file_counter = 0
    for i in range(2):
        tracker = EmissionsTracker(project_name=f"musicAudioLDM-inference_batch5", tracking_mode="process",
                                   gpu_ids="0",
                                   output_dir="../../results/batch_size/musicAudioLDM/batch5",
                                   output_file=f"musicAudioLDM-emissions-batch5-n{i}-run{k+1}.csv", allow_multiple_runs=True)
        tracker.start_task(f"Inference emissions with 5 waveforms per prompt")

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
                f"genaudios/musicAudioLDM/musicAudioLDM-batch5-n{file_counter}.wav",
                rate=16000, data=audio[x])
            file_counter += 1
        print(model_emissions)


for k in range(runs):
    file_counter = 0
    for i in range(5):
        tracker = EmissionsTracker(project_name=f"musicAudioLDM-inference_batch2", tracking_mode="process",
                                   gpu_ids="0",
                                   output_dir="../../results/batch_size/musicAudioLDM/batch2",
                                   output_file=f"musicAudioLDM-emissions-batch2-n{i}-run{k+1}.csv", allow_multiple_runs=True)
        tracker.start_task(f"Inference emissions with 2 waveforms per prompt")

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
            f"genaudios/musicAudioLDM/musicAudioLDM-batch2-n{file_counter}.wav",
            rate=16000, data=audio[x])
            file_counter += 1
        print(model_emissions)

for k in range(runs):
    for i in range(10):
        tracker = EmissionsTracker(project_name=f"musicAudioLDM-inference_batch1", tracking_mode="process",
                                   gpu_ids="0",
                                   output_dir="../../results/batch_size/musicAudioLDM/batch1",
                                   output_file=f"musicAudioLDM-emissions-batch1-n{i}-run{k+1}.csv", allow_multiple_runs=True)
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
            f"genaudios/musicAudioLDM/musicAudioLDM-batch1-n{i}.wav",
            rate=16000, data=audio[0])
        print(model_emissions)


emissions_base = glob.glob('../../results/batch_size/musicAudioLDM/emissions_base_*')
for f in emissions_base:
    os.remove(f)

emissions_base = glob.glob('../../results/batch_size/musicAudioLDM/batch5/emissions_base_*')
for f in emissions_base:
    os.remove(f)

emissions_base = glob.glob('../../results/batch_size/musicAudioLDM/batch2/emissions_base_*')
for f in emissions_base:
    os.remove(f)

emissions_base = glob.glob('../../results/batch_size/musicAudioLDM/batch1/emissions_base_*')
for f in emissions_base:
    os.remove(f)

print("Done!")
