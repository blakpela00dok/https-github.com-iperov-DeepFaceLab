import pickle
import json 
import os 

dictionary = {
        '4' : '\n',
        '5' : '0',
        '2' : '1',
        '3' : '4',
        '1' : '0',
        'Output image format':'png', 
        'NoInteractiveMode':'y', 
        'Which GPU indexes to choose?': '0',
        'Face type': 'wf',
        'Max number of faces from image' : '1',
        'Image size' : '512', 
        'Jpeg quality' : '90',
        'Write debug images to aligned_debug?': 'n', 
        'Autobackup every N hour':'2', 
        'Write preview history' : 'n', 
        'Flip SRC faces randomly':'n', 
        'Flip DST faces randomly':'n', 
        'Batch_size': '4', 
        'Eyes and mouth priority':'n', 
        'Uniform yaw distribution of samples':'y', 
        'Blur out mask':'n', 
        'Place models and optimizer on GPU' : 'y', 
        'Use AdaBelief optimizer?' : 'y', 
        'Use learning rate dropout' : 'n', 
        'Enable random warp of samples' : 'y', 
        'Random hue/saturation/light intensity' : '0.0', 
        'GAN power' : '0.0', 
        'Face style power' : '0.0', 
        'Background style power': '0.0', 
        'Color transfer for src faceset' : 'lct', 
        'Enable gradient clipping': 'n',
        'Enable pretraining mode' : 'n', 
        'Use interactive merger?':'n', 
        'Number of workers?':'8', 
        'Use saved session?':'n', 
        'Bitrate of output file in MB/s' : '16', 
        'Choose erode mask modifier': '80', 
        'Choose blur mask modifier' : '100', 
        'Choose motion blur power' : '0',
        'Choose output face scale modifier' : '0',
        'Choose super resolution power' : '0',
        'Choose image degrade by denoise power' : '0',
        'Choose image degrade by bicubic rescale power' : '0',
        'Degrade color power of final image' : '0',
        'Color transfer to predicted face' : 'rct',
        'Press enter in 2 seconds to override model settings.' : 'y',
}
cmd = 'mkdir DeepFaceLab_Linux/workspace/interact'
os.system(cmd)
with open('DeepFaceLab_Linux/workspace/interact/interact_dict.json', 'w') as handle:
	json.dump(dictionary, handle)

#with open('DeepFaceLab_Linux/workspace/interact/interact_dict.pkl', 'rb') as handle:
#	d = pickle.load(handle)
#
#print(d['5'])
