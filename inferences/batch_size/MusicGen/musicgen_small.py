from transformers import AutoProcessor, MusicgenForConditionalGeneration
from codecarbon import EmissionsTracker
from itertools import chain
import scipy
import os
import glob

emissions = glob.glob('../../../results/batch_size/MusicGen/Small/*')
emissions1 = glob.glob('../../../results/batch_size/MusicGen/Small/batch1/*')
emissions2 = glob.glob('../../../results/batch_size/MusicGen/Small/batch2/*')
emissions5 = glob.glob('../../../results/batch_size/batchexp/MusicGen/Small/batch5/*')
genaudios = glob.glob('../genaudios/MusicGen/Small/*')
for f in chain(emissions, emissions1, emissions2, emissions5, genaudios):
    if os.path.isfile(f):
    	os.remove(f)

processor = AutoProcessor.from_pretrained("facebook/musicgen-small", device='cuda')
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
model = model.to('cuda:0')

inputs = processor(
    text=["An organ is playing chords and melodies while a male voice is singing soulfully."]*10,
    padding=True, return_tensors="pt"
).to("cuda:0")

runs = 5

for k in range(runs):

    tracker = EmissionsTracker(project_name=f"MusicGenSmall-inference_batch10", tracking_mode="process", gpu_ids="0",
                                output_dir="../../../results/batch_size/MusicGen/Small",
                                output_file=f"MusicGenSmall-emissions-batch10-run{k+1}.csv", allow_multiple_runs=True)
    tracker.start_task(f"Inference emissions with 10 waveforms per prompt")

    try:

        audio_values = model.generate(**inputs, max_new_tokens=500)

        model_emissions = tracker.stop_task()

    finally:
        _ = tracker.stop()

    sampling_rate = model.config.audio_encoder.sampling_rate
    for i in range(10):
        out_path = f"../genaudios/MusicGen/Small/MusicGenSmall-batch10-n{i}.wav"
        scipy.io.wavfile.write(out_path, rate=sampling_rate, data=audio_values[i, 0].cpu().numpy())

inputs = processor(
    text=["An organ is playing chords and melodies while a male voice is singing soulfully."]*5,
    padding=True, return_tensors="pt"
).to("cuda:0")

for k in range(runs):
    file_counter = 0
    for i in range(2):
        
        tracker = EmissionsTracker(project_name=f"MusicGenSmall-inference_batch5", tracking_mode="process", gpu_ids="0",
                                    output_dir="../../../results/batch_size/MusicGen/Small/batch5",
                                    output_file=f"MusicGenSmall-emissions-batch5-n{i}-run{k+1}.csv", allow_multiple_runs=True)
        tracker.start_task(f"Inference emissions with 5 waveforms per prompt")

        try:

            audio_values = model.generate(**inputs, max_new_tokens=500)

            model_emissions = tracker.stop_task()

        finally:
            _ = tracker.stop()

        sampling_rate = model.config.audio_encoder.sampling_rate
        for i in range(5):
            out_path = f"../genaudios/MusicGen/Small/MusicGenSmall-batch5-n{file_counter}.wav"
            scipy.io.wavfile.write(out_path, rate=sampling_rate, data=audio_values[i, 0].cpu().numpy())
            file_counter += 1

inputs = processor(
    text=["An organ is playing chords and melodies while a male voice is singing soulfully."]*2,
    padding=True, return_tensors="pt"
).to("cuda:0")

for k in range(runs):
    file_counter = 0
    for i in range(5):
        
        tracker = EmissionsTracker(project_name=f"MusicGenSmall-inference_batch2", tracking_mode="process", gpu_ids="0",
                                    output_dir="../../../results/batch_size/MusicGen/Small/batch2",
                                    output_file=f"MusicGenSmall-emissions-batch2-n{i}-run{k+1}.csv", allow_multiple_runs=True)
        tracker.start_task(f"Inference emissions with 2 waveforms per prompt")

        try:

            audio_values = model.generate(**inputs, max_new_tokens=500)

            model_emissions = tracker.stop_task()

        finally:
            _ = tracker.stop()

        sampling_rate = model.config.audio_encoder.sampling_rate
        for i in range(2):
            out_path = f"../genaudios/MusicGen/Small/MusicGenSmall-batch2-n{file_counter}.wav"
            scipy.io.wavfile.write(out_path, rate=sampling_rate, data=audio_values[i, 0].cpu().numpy())
            file_counter += 1

inputs = processor(
    text=["An organ is playing chords and melodies while a male voice is singing soulfully."],
    padding=True, return_tensors="pt"
).to("cuda:0")

for k in range(runs):
    for i in range(10):

        tracker = EmissionsTracker(project_name=f"MusicGenSmall-inference_batch1", tracking_mode="process", gpu_ids="0",
                                    output_dir="../../../results/batch_size/MusicGen/Small/batch1",
                                    output_file=f"MusicGenSmall-emissions-batch1-n{i}-run{k+1}.csv", allow_multiple_runs=True)
        tracker.start_task(f"Inference emissions with 1 waveforms per prompt")

        try:

            audio_values = model.generate(**inputs, max_new_tokens=500)

            model_emissions = tracker.stop_task()

        finally:
            _ = tracker.stop()

        sampling_rate = model.config.audio_encoder.sampling_rate
        scipy.io.wavfile.write(f"../genaudios/MusicGen/Small/MusicGenSmall-batch1-n{i}.wav",
                        rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())


emissions_base = glob.glob('../../../results/batch_size/MusicGen/Small/emissions_base_*')
for f in emissions_base:
    os.remove(f)

emissions_base = glob.glob('../../../results/batch_size/MusicGen/Small/batch5/emissions_base_*')
for f in emissions_base:
    os.remove(f)

emissions_base = glob.glob('../../../results/batch_size/MusicGen/Small/batch2/emissions_base_*')
for f in emissions_base:
    os.remove(f)

emissions_base = glob.glob('../../../results/batch_size/MusicGen/Small/batch1/emissions_base_*')
for f in emissions_base:
    os.remove(f)

print("Done!")
