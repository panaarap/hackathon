import ifxdaq
import processing
import numpy as np
from PIL import Image
import statistics


#print(ifxdaq.__version__)
from ifxdaq.sensor.radar_ifx import RadarIfxAvian
import matplotlib.pyplot as plot




config_file = "radar_configs/RadarIfxBGT60.json"
number_of_frames = 1500

## Run this to understand the current radar settings better
import json
# with open(config_file) as json_file:
#     c = json.load(json_file)["device_config"]["fmcw_single_shape"]
#     chirp_duration = c["num_samples_per_chirp"]/c['sample_rate_Hz']
#     frame_duration = (chirp_duration + c['chirp_repetition_time_s']) * c['num_chirps_per_frame']
#     print("With the current configuration, the radar will send out " + str(c['num_chirps_per_frame']) + \
#           ' signals with varying frequency ("chirps") between ' + str(c['start_frequency_Hz']/1e9) + " GHz and " + \
#           str(c['end_frequency_Hz']/1e9) + " GHz.")
#     print('Each chirp will consist of ' + str(c["num_samples_per_chirp"]) + ' ADC measurements of the IF signal ("samples").')
#     print('A chirp takes ' + str(chirp_duration*1e6) + ' microseconds and the delay between the chirps is ' + str(c['chirp_repetition_time_s']*1e6) +' microseconds.')
#     print('With a total frame duration of ' + str(frame_duration*1e3) + ' milliseconds and a delay of ' + str(c['frame_repetition_time_s']*1e3) + ' milliseconds between the frame we get a frame rate of ' + str(1/(frame_duration + c['frame_repetition_time_s'])) + ' radar frames per second.')

raw_data = []
sample = []
with RadarIfxAvian(config_file) as device:  # Initialize the radar with configurations
    print(device.device_id)
    for i_frame, frame in enumerate(device):  # Loop through the frames coming from the radar

        raw_data.append(np.squeeze(frame['radar'].data / (4095.0)))  # Dividing by 4095.0 to scale the data
        if (i_frame == number_of_frames - 1):
            data = np.asarray(raw_data)
            range_doppler_map = processing.processing_rangeDopplerData(data)
            # del data
            break

mean_antenna = []

for s in range(number_of_frames):
             mean_antenna.append((range_doppler_map[s][0][:][:]+range_doppler_map[s][1][:][:]+range_doppler_map[s][2][:][:])/3)




for s in range(1,number_of_frames):
            sample.append(range_doppler_map[s][:][:][:] - range_doppler_map[s][:][:][:])


mean_antenna = np.array(mean_antenna)
sample = np.array(sample)
print(sample.shape)
print("(frame_index, receiver_index, chirp_index, sample_index)")
print(data.shape)
# print(range_doppler_map.shape)
# path = "C:\Users\jimmy\PycharmProjects\hackathon\examples\zerop"
for i in range(number_of_frames):
    np.save("./processed/d.three"+str(i)+".npy", range_doppler_map[i])
    np.save("./raw/d.three" + str(i) + ".npy", raw_data[i])
    extended=np.concatenate((range_doppler_map[i][0],range_doppler_map[i][1],range_doppler_map[i][2]))
    # print(extended.shape)
    np.save("./extended/d.three" + str(i) + ".npy", extended)
    if (i%50 == 0):
        print(i)





#print(range_doppler_map)
fig, axs = plot.subplots(3, 5,figsize=(15,10),sharex=True, sharey=True)
fig.suptitle('Range-Doppler Plot')
# medians = [np.median([range_doppler_map[0][1][:][:],range_doppler_map[0][1][:][:],range_doppler_map[0][1][:][:])]


for i in range(3):
    for j in range(5):
        axs[i, j].imshow(np.abs(raw_data)[j,i,:,:])
        axs[i, j].set_aspect('equal')

plot.subplots_adjust(hspace=0)



plot.show()