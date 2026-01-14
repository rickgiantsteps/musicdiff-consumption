from audiocraft.models import MAGNeT
from codecarbon import EmissionsTracker
from audiocraft.data.audio import audio_write
from itertools import chain
import glob
import os

emissions = glob.glob('../../../results/batch_size/Magnet/Small/*')
emissions1 = glob.glob('../../../results/batch_size/Magnet/Small/batch1/*')
emissions2 = glob.glob('../../../results/batch_size/Magnet/Small/batch2/*')
emissions5 = glob.glob('../../../results/batch_size/Magnet/Small/batch5/*')
genaudios = glob.glob('../genaudios/Magnet/Small/*')
for f in chain(emissions, emissions1, emissions2, emissions5, genaudios):
    if os.path.isfile(f):
    	os.remove(f)

model = MAGNeT.get_pretrained("facebook/magnet-small-10secs")

descriptions=["An organ is playing chords and melodies while a male voice is singing soulfully."]*10

runs = 5

for k in range(runs):

    tracker = EmissionsTracker(project_name=f"MagnetSmall-inference_batch10", tracking_mode="process", gpu_ids="0",
                                output_dir="../../../results/batch_size/Magnet/Small",
                                output_file=f"MagnetSmall-emissions-batch10-run{k+1}.csv", allow_multiple_runs=True)
    tracker.start_task(f"Inference emissions with 10 waveforms per prompt")

    try:

        wav = model.generate(descriptions)

        model_emissions = tracker.stop_task()

    finally:
        _ = tracker.stop()

    for idx, one_wav in enumerate(wav):
       audio_write(f"../genaudios/Magnet/Small/MagnetSmall-batch10-n{idx}", one_wav.cpu(), model.sample_rate, strategy="loudness")

descriptions=["An organ is playing chords and melodies while a male voice is singing soulfully."]*5

for k in range(runs):
    file_counter = 0
    for i in range(2):
        tracker = EmissionsTracker(project_name=f"MagnetSmall-inference_batch5", tracking_mode="process", gpu_ids="0",
                            output_dir="../../../results/batch_size/Magnet/Small/batch5",
                            output_file=f"MagnetSmall-emissions-batch5-n{i}-run{k+1}.csv", allow_multiple_runs=True)
        tracker.start_task(f"Inference emissions with 5 waveforms per prompt")

        try:

            wav = model.generate(descriptions)

            model_emissions = tracker.stop_task()

        finally:
            _ = tracker.stop()

        for idx, one_wav in enumerate(wav):
            audio_write(f"../genaudios/Magnet/Small/MagnetSmall-batch5-n{file_counter}", one_wav.cpu(), model.sample_rate, strategy="loudness")
            file_counter += 1

descriptions=["An organ is playing chords and melodies while a male voice is singing soulfully."]*2

for k in range(runs):
    file_counter = 0
    for i in range(5):

        tracker = EmissionsTracker(project_name=f"MagnetSmall-inference_batch2", tracking_mode="process", gpu_ids="0",
                            output_dir="../../../results/batch_size/Magnet/Small/batch2",
                            output_file=f"MagnetSmall-emissions-batch2-n{i}-run{k+1}.csv", allow_multiple_runs=True)
        tracker.start_task(f"Inference emissions with 2 waveforms per prompt")

        try:

            wav = model.generate(descriptions)

            model_emissions = tracker.stop_task()

        finally:
            _ = tracker.stop()

        for idx, one_wav in enumerate(wav):
            audio_write(f"../genaudios/Magnet/Small/MagnetSmall-batch2-n{file_counter}", one_wav.cpu(), model.sample_rate, strategy="loudness")
            file_counter += 1

descriptions=["An organ is playing chords and melodies while a male voice is singing soulfully."]

for k in range(runs):
    for i in range(10):
        tracker = EmissionsTracker(project_name=f"MagnetSmall-inference_batch1", tracking_mode="process", gpu_ids="0",
                                output_dir="../../../results/batch_size/Magnet/Small/batch1",
                                output_file=f"MagnetSmall-emissions-batch1-n{i}-run{k+1}.csv", allow_multiple_runs=True)
        tracker.start_task(f"Inference emissions with 10 waveforms per prompt")

        try:

            wav = model.generate(descriptions)

            model_emissions = tracker.stop_task()

        finally:
            _ = tracker.stop()

        for idx, one_wav in enumerate(wav):
            audio_write(f"../genaudios/Magnet/Small/MagnetSmall-batch1-n{i}", one_wav.cpu(), model.sample_rate, strategy="loudness")


emissions_base = glob.glob('../../../results/batch_size/Magnet/Small/emissions_base_*')
for f in emissions_base:
    os.remove(f)

emissions_base = glob.glob('../../../results/batch_size/Magnet/Small/batch5/emissions_base_*')
for f in emissions_base:
    os.remove(f)

emissions_base = glob.glob('../../../results/batch_size/Magnet/Small/batch2/emissions_base_*')
for f in emissions_base:
    os.remove(f)

emissions_base = glob.glob('../../../results/batch_size/Magnet/Small/batch1/emissions_base_*')
for f in emissions_base:
    os.remove(f)

print("Done!")
