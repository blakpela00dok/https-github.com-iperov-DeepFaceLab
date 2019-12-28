import numpy as np

from nnlib import nnlib
from models import ModelBase
from facelib import FaceType
from samplelib import *
from interact import interact as io

class Model(ModelBase):

    #override
    def onInitializeOptions(self, is_first_run, ask_override):
        if is_first_run or ask_override:
            def_pixel_loss = self.options.get('pixel_loss', False)
            self.options['pixel_loss'] = io.input_bool ("Use pixel loss? (y/n, ?:help skip: n/default ) : ", def_pixel_loss, help_message="Pixel loss may help to enhance fine details and stabilize face color. Use it only if quality does not improve over time.")
        else:
            self.options['pixel_loss'] = self.options.get('pixel_loss', False)

    #override
    def onInitialize(self):
        exec(nnlib.import_all(), locals(), globals())
        self.set_vram_batch_requirements( {4.5:4} )

        ae_input_layer = Input(shape=(128, 128, 3))
        mask_layer = Input(shape=(128, 128, 1)) #same as output

        self.encoder, self.decoder, self.inter_B, self.inter_AB = self.Build(ae_input_layer)

        if not self.is_first_run():
            weights_to_load = [  [self.encoder,  'encoder.h5'],
                                 [self.decoder,  'decoder.h5'],
                                 [self.inter_B,  'inter_B.h5'],
                                 [self.inter_AB, 'inter_AB.h5']
                              ]
            self.load_weights_safe(weights_to_load)

        code = self.encoder(ae_input_layer)
        AB = self.inter_AB(code)
        B = self.inter_B(code)
        rec_src = self.decoder(Concatenate()([AB, AB]))
        rec_dst = self.decoder(Concatenate()([B, AB]))
        self.autoencoder_src = Model([ae_input_layer,mask_layer], rec_src )
        self.autoencoder_dst = Model([ae_input_layer,mask_layer], rec_dst )

        self.autoencoder_src.compile(optimizer=Adam(lr=5e-5, beta_1=0.5, beta_2=0.999), loss=[DSSIMMSEMaskLoss(mask_layer, is_mse=self.options['pixel_loss']), 'mse'] )
        self.autoencoder_dst.compile(optimizer=Adam(lr=5e-5, beta_1=0.5, beta_2=0.999), loss=[DSSIMMSEMaskLoss(mask_layer, is_mse=self.options['pixel_loss']), 'mse'] )

        self.convert = K.function([ae_input_layer],rec_src)


        if self.is_training_mode:
            t = SampleProcessor.Types
            output_sample_types=[ { 'types': (t.IMG_WARPED_TRANSFORMED, t.FACE_TYPE_FULL, t.MODE_BGR), 'resolution':128},
                                  { 'types': (t.IMG_TRANSFORMED, t.FACE_TYPE_FULL, t.MODE_BGR), 'resolution':128},
                                  { 'types': (t.IMG_TRANSFORMED, t.FACE_TYPE_FULL, t.MODE_M), 'resolution':128} ]

            self.set_training_data_generators ([
                    SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip, scale_range=np.array([-0.05, 0.05]) ),
                        output_sample_types=output_sample_types),

                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip),
                        output_sample_types=output_sample_types)
                ])

    #override
    def get_model_filename_list(self):
        return [[self.encoder, 'encoder.h5'],
                [self.decoder, 'decoder.h5'],
                [self.inter_B, 'inter_B.h5'],
                [self.inter_AB, 'inter_AB.h5']]

    #override
    def onSave(self):
        self.save_weights_safe( self.get_model_filename_list() )

    #override
    def onTrainOneIter(self, sample, generators_list):
        warped_src, target_src, target_src_mask = sample[0]
        warped_dst, target_dst, target_dst_mask = sample[1]

        loss_src = self.autoencoder_src.train_on_batch( [warped_src, target_src_mask], [target_src, target_src_mask] )
        loss_dst = self.autoencoder_dst.train_on_batch( [warped_dst, target_dst_mask], [target_dst, target_dst_mask] )

        return ( ('loss_src', loss_src[0]), ('loss_dst', loss_dst[0]) )


    #override
    def onGetPreview(self, sample):
        test_A   = sample[0][1][0:4] #first 4 samples
        test_A_m = sample[0][2][0:4] #first 4 samples
        test_B   = sample[1][1][0:4]
        test_B_m = sample[1][2][0:4]

        AA, mAA = self.autoencoder_src.predict([test_A, test_A_m])
        AB, mAB = self.autoencoder_src.predict([test_B, test_B_m])
        BB, mBB = self.autoencoder_dst.predict([test_B, test_B_m])

        mAA = np.repeat ( mAA, (3,), -1)
        mAB = np.repeat ( mAB, (3,), -1)
        mBB = np.repeat ( mBB, (3,), -1)

        st = []
        for i in range(0, len(test_A)):
            st.append ( np.concatenate ( (
                test_A[i,:,:,0:3],
                AA[i],
                #mAA[i],
                test_B[i,:,:,0:3],
                BB[i],
                #mBB[i],
                AB[i],
                #mAB[i]
                ), axis=1) )

        return [ ('LIAEF128', np.concatenate ( st, axis=0 ) ) ]

    def predictor_func (self, face=None, dummy_predict=False):
        if dummy_predict:
            self.convert ([ np.zeros ( (1, 128, 128, 3), dtype=np.float32 ) ])
        else:
            x, mx = self.convert ( [ face[np.newaxis,...] ] )
            return x[0], mx[0][...,0]

    #override
    def get_ConverterConfig(self):
        import converters
        return self.predictor_func, (128,128,3), converters.ConverterConfigMasked(face_type=FaceType.FULL, default_mode='seamless')

    def Build(self, input_layer):
        exec(nnlib.code_import_all, locals(), globals())

        def downscale (dim):
            def func(x):
                return LeakyReLU(0.1)(Conv2D(dim, 5, strides=2, padding='same')(x))
            return func

        def upscale (dim):
            def func(x):
                return PixelShuffler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func

        def Encoder():
            x = input_layer
            x = downscale(128)(x)
            x = downscale(256)(x)
            x = downscale(512)(x)
            x = downscale(1024)(x)
            x = Flatten()(x)
            return Model(input_layer, x)

        def Intermediate():
            input_layer = Input(shape=(None, 8 * 8 * 1024))
            x = input_layer
            x = Dense(256)(x)
            x = Dense(8 * 8 * 512)(x)
            x = Reshape((8, 8, 512))(x)
            x = upscale(512)(x)
            return Model(input_layer, x)

        def Decoder():
            input_ = Input(shape=(16, 16, 1024))
            x = input_
            x = upscale(512)(x)
            x = upscale(256)(x)
            x = upscale(128)(x)
            x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)

            y = input_  #mask decoder
            y = upscale(512)(y)
            y = upscale(256)(y)
            y = upscale(128)(y)
            y = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid' )(y)

            return Model(input_, [x,y])

        return Encoder(), Decoder(), Intermediate(), Intermediate()
