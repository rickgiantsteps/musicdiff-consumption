from codecarbon import EmissionsTracker
import soundfile as sf
from tango import Tango
import torch
import os
import glob

emissions = glob.glob('../../results/batch_size/musicTango/*')
emissions1 = glob.glob('../../results/batch_size/musicTango/batch1/*')
emissions2 = glob.glob('../../results/batch_size/musicTango/batch2/*')
emissions5 = glob.glob('../../results/batch_size/musicTango/batch5/*')
genaudios = glob.glob('genaudios/musicSAO/*')
for f in chain(emissions, emissions1, emissions2, emissions5, genaudios):
    if os.path.isfile(f):
    	os.remove(f)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tango = Tango("declare-lab/tango-music-af-ft-mc")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tango.model = tango.model.to(device)  # Move the model to GPU

prompt = "An organ is playing chords and melodies while a male voice is singing soulfully."

runs = 5

for k in range(runs):

    tracker = EmissionsTracker(project_name=f"musicTango-inference_batch10", tracking_mode="process",
                                gpu_ids="0",
                                output_dir="../../results/batch_size/musicTango",
                                output_file=f"musicTango-emissions-batch10-run{k+1}.csv", allow_multiple_runs=True)
    tracker.start_task(f"Inference emissions with 10 waveforms per prompt")

    try:

        audio = tango.generate_for_batch(list(map(lambda _: prompt, range(10))), steps=100)

        model_emissions = tracker.stop_task()

    finally:
        _ = tracker.stop()

    for x in range(10):
        sf.write(f"genaudios/musicSAO/musicTango-batch10-n{x}.wav",
                audio[x], samplerate=16000)

    print(model_emissions)

for k in range(runs):
    file_counter = 0
    for i in range(2):
        tracker = EmissionsTracker(project_name=f"musicTango-inference_batch5", tracking_mode="process",
                                   gpu_ids="0",
                                   output_dir="../../results/batch_size/musicTango/batch5",
                                   output_file=f"musicTango-emissions-batch5-n{i}-run{k+1}.csv", allow_multiple_runs=True)
        tracker.start_task(f"Inference emissions with 5 waveforms per prompt")

        try:

            audio = tango.generate_for_batch(list(map(lambda _: prompt, range(5))), steps=100)

            model_emissions = tracker.stop_task()

        finally:
            _ = tracker.stop()

        for x in range(5):
            sf.write(f"genaudios/musicSAO/musicTango-batch5-n{file_counter}.wav",
                    audio[x], samplerate=16000)
            file_counter += 1

    print(model_emissions)

for k in range(runs):
    file_counter = 0
    for i in range(5):
        tracker = EmissionsTracker(project_name=f"musicTango-inference_batch2", tracking_mode="process",
                                   gpu_ids="0",
                                   output_dir="../../results/batch_size/musicTango/batch2",
                                   output_file=f"musicTango-emissions-batch2-n{i}-run{k+1}.csv", allow_multiple_runs=True)
        tracker.start_task(f"Inference emissions with 2 waveforms per prompt")

        try:

            audio = tango.generate_for_batch(list(map(lambda _: prompt, range(2))), steps=100)

            model_emissions = tracker.stop_task()

        finally:
            _ = tracker.stop()

        for x in range(2):
            sf.write(f"genaudios/musicSAO/musicTango-batch2-n{file_counter}.wav",
                    audio[x], samplerate=16000)
            file_counter += 1

    print(model_emissions)

for k in range(runs):
    for i in range(10):
        tracker = EmissionsTracker(project_name=f"musicTango-inference_batch1", tracking_mode="process",
                                   gpu_ids="0",
                                   output_dir="../../results/batch_size/musicTango/batch1",
                                   output_file=f"musicTango-emissions-batch1-n{i}-run{k+1}.csv", allow_multiple_runs=True)
        tracker.start_task(f"Inference emissions with 1 waveforms per prompt")

        try:

            audio = tango.generate(prompt, steps=100)

            model_emissions = tracker.stop_task()

        finally:
            _ = tracker.stop()

        sf.write(f"genaudios/musicSAO/musicTango-batch1-n{i}.wav",
                audio, samplerate=16000)

    print(model_emissions)

emissions_base = glob.glob('../../results/batch_size/musicTango/emissions_base_*')
for f in emissions_base:
    os.remove(f)

emissions_base = glob.glob('../../results/batch_size/musicTango/batch5/emissions_base_*')
for f in emissions_base:
    os.remove(f)

emissions_base = glob.glob('../../results/batch_size/musicTango/batch2/emissions_base_*')
for f in emissions_base:
    os.remove(f)

emissions_base = glob.glob('../../results/batch_size/musicTango/batch1/emissions_base_*')
for f in emissions_base:
    os.remove(f)

print("Done!")
