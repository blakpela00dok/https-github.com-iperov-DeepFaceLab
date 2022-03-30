import pickle

dictionary = {
        '4' : '\n',
        'Output image format':'png', 
        'Which GPU indexes to choose?': '0',
'Face type': 'wf',
'Max number of faces from image' : '1',
'Image size' : '512', 
'Jpeg quality' : '90',
'Write debug images to aligned_debug?': 'False', 
'Autobackup every N hour':'2', 
'Write preview history' : 'False', 
'Flip SRC faces randomly':'False', 
'Flip DST faces randomly':'False', 
'Batch_size': '4', 
'Eyes and mouth priority':'False', 
'Uniform yaw distribution of samples':'True', 
'Blur out mask':'False', 
'Place models and optimizer on GPU' : 'True', 
'Use AdaBelief optimizer?' : 'True', 
'Use learning rate dropout' : 'False', 
'Enable random warp of samples' : 'True', 
'Random hue/saturation/light intensity' : '0.0', 
'GAN power' : '0.0', 
'Face style power' : '0.0', 
'Background style power': '0.0', 
'Color transfer for src faceset' : 'lct', 
'Enable gradient clipping': 'False',
'Enable pretraining mode' : 'False', 
'Use interactive merger?':'False', 
'Number of workers?':'8', 
'Use saved session?':'False', 
'Bitrate of output file in MB/s' : '16', 
'Choose erode mask modifier': '0.0', 
'Choose blur mask modifier' : '0.0', 
'Choose motion blur power' : '0',
'Choose output face scale modifier' : '0',
'Choose super resolution power' : '0',
'Choose image degrade by denoise power' : '0',
'Choose image degrade by bicubic rescale power' : '0',
'Degrade color power of final image' : '0',
'Color transfer to predicted face' : 'rct',
}
with open('/home/deepfake/interact_dict.pkl', 'wb') as handle:
	pickle.dump(dictionary, handle, protocol=4)

with open('/home/deepfake/interact_dict.pkl', 'rb') as handle:
	d = pickle.load(handle)

print(d['Color transfer to predicted face'])
