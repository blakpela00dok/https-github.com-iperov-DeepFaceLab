import numpy as np

from nnlib import nnlib, TernausNet
from models import ModelBase
from facelib import FaceType
from samplelib import *
from interact import interact as io

class Model(ModelBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, 
                            ask_enable_autobackup=False,
                            ask_write_preview_history=False, 
                            ask_target_iter=False,
                            ask_random_flip=False)
    
    #override
    def onInitializeOptions(self, is_first_run, ask_override):            
        default_face_type = 'f'
        if is_first_run:
            self.options['face_type'] = io.input_str ("Half or Full face? (h/f, ?:help skip:f) : ", default_face_type, ['h','f'], help_message="").lower()
        else:
            self.options['face_type'] = self.options.get('face_type', default_face_type)
     
    #override
    def onInitialize(self):
        exec(nnlib.import_all(), locals(), globals())
        self.set_vram_batch_requirements( {1.5:4, 11:48} )

        self.resolution = 256
        self.face_type = FaceType.FULL if self.options['face_type'] == 'f' else FaceType.HALF

        model_name = 'FANSeg'
        self.fan_seg = TernausNet(model_name, self.resolution, 
                                            FaceType.toString(self.face_type), 
                                            load_weights=not self.is_first_run(),
                                            weights_file_root=self.get_model_root_path(),
                                            training=True)

        if self.is_training_mode:
            t = SampleProcessor.Types
            face_type = t.FACE_TYPE_FULL if self.options['face_type'] == 'f' else t.FACE_TYPE_HALF
            
            self.set_training_data_generators ([    
                    SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size, 
                            sample_process_options=SampleProcessor.Options(random_flip=True), 
                            output_sample_types=[ { 'types': (t.IMG_WARPED_TRANSFORMED, face_type, t.MODE_BGR_RANDOM_HSV_SHIFT), 'resolution' : self.resolution, 'motion_blur':(25, 5),  'gaussian_blur':(25,5), 'border_replicate':False},
                                                  { 'types': (t.IMG_WARPED_TRANSFORMED, face_type, t.MODE_M), 'resolution': self.resolution },
                                                ]),
                                                
                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size, 
                            sample_process_options=SampleProcessor.Options(random_flip=True ), 
                            output_sample_types=[ { 'types': (t.IMG_TRANSFORMED , face_type, t.MODE_BGR_RANDOM_HSV_SHIFT), 'resolution' : self.resolution},
                                                ])
                                               ])
                
    #override
    def onSave(self):        
        self.fan_seg.save_weights()
        
    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        target_src, target_src_mask = generators_samples[0]

        loss = self.fan_seg.train( target_src, target_src_mask )

        return ( ('loss', loss), )
        
    #override
    def onGetPreview(self, sample):
        test_A      = sample[0][0][0:4] #first 4 samples
        test_Am     = sample[0][1][0:4] #first 4 samples
        test_B      = sample[1][0][0:4] #first 4 samples
        
        
        mAA = self.fan_seg.extract(test_A)
        mBB = self.fan_seg.extract(test_B)

        test_Am = np.repeat ( test_Am, (3,), -1)
        mAA = np.repeat ( mAA, (3,), -1)
        mBB = np.repeat ( mBB, (3,), -1)
        
        st = []
        for i in range(0, len(test_A)):
            st.append ( np.concatenate ( (
                test_A[i,:,:,0:3],
                test_Am[i],
                mAA[i],
                test_A[i,:,:,0:3]*mAA[i],
                ), axis=1) )
                
        st2 = []
        for i in range(0, len(test_B)):
            st2.append ( np.concatenate ( (
                test_B[i,:,:,0:3],                
                mBB[i],
                test_B[i,:,:,0:3]*mBB[i],
                ), axis=1) )
                
        return [ ('training data', np.concatenate ( st, axis=0 ) ),
                 ('evaluating data', np.concatenate ( st2, axis=0 ) ),
                 ]
