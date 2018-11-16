from models import ConverterBase
from facelib import LandmarksProcessor
from facelib import FaceType
import cv2
import numpy as np
from utils import image_utils

'''
predictor: 
    input:  [predictor_input_size, predictor_input_size, BGRA]
    output: [predictor_input_size, predictor_input_size, BGRA]
'''

class ConverterMasked(ConverterBase):

    #override
    def __init__(self,  predictor,
                        predictor_input_size=0, 
                        output_size=0, 
                        face_type=FaceType.FULL, 
                        erode_mask = True, 
                        blur_mask = True,
                        clip_border_mask_per = 0,
                        masked_hist_match = False, 
                        mode='seamless', 
                        erode_mask_modifier=0, 
                        blur_mask_modifier=0,                         
                        **in_options):
                        
        super().__init__(predictor)
         
        self.predictor_input_size = predictor_input_size
        self.output_size = output_size
        self.face_type = face_type        
        self.erode_mask = erode_mask
        self.blur_mask = blur_mask
        self.clip_border_mask_per = clip_border_mask_per
        self.masked_hist_match = masked_hist_match
        self.mode = mode
        self.erode_mask_modifier = erode_mask_modifier
        self.blur_mask_modifier = blur_mask_modifier
        
        if self.erode_mask_modifier != 0 and not self.erode_mask:
            print ("Erode mask modifier not used in this model.")
            
        if self.blur_mask_modifier != 0 and not self.blur_mask:
            print ("Blur modifier not used in this model.")
            
    #override
    def get_mode(self):
        return ConverterBase.MODE_FACE
        
    #override
    def dummy_predict(self):
        self.predictor ( np.zeros ( (self.predictor_input_size,self.predictor_input_size,4), dtype=np.float32 ) )
        
    #override
    def convert_face (self, img_bgr, img_face_landmarks, debug, alpha, transfercolor):        
        if debug:        
            debugs = [img_bgr.copy()]

        img_size = img_bgr.shape[1], img_bgr.shape[0]

        img_face_mask_a = LandmarksProcessor.get_image_hull_mask (img_bgr, img_face_landmarks)
        
        face_mat = LandmarksProcessor.get_transform_mat (img_face_landmarks, self.output_size, face_type=self.face_type)
        dst_face_bgr      = cv2.warpAffine( img_bgr        , face_mat, (self.output_size, self.output_size), flags=cv2.INTER_LANCZOS4 )
        dst_face_mask_a_0 = cv2.warpAffine( img_face_mask_a, face_mat, (self.output_size, self.output_size), flags=cv2.INTER_LANCZOS4 )

        predictor_input_bgr      = cv2.resize (dst_face_bgr,      (self.predictor_input_size,self.predictor_input_size))
        predictor_input_mask_a_0 = cv2.resize (dst_face_mask_a_0, (self.predictor_input_size,self.predictor_input_size))
        predictor_input_mask_a   = np.expand_dims (predictor_input_mask_a_0, -1) 
        
        predicted_bgra = self.predictor ( np.concatenate( (predictor_input_bgr, predictor_input_mask_a), -1) )

        prd_face_bgr      = np.clip (predicted_bgra[:,:,0:3], 0, 1.0 )
        prd_face_mask_a_0 = np.clip (predicted_bgra[:,:,3], 0.0, 1.0)
        prd_face_mask_a_0[ prd_face_mask_a_0 < 0.001 ] = 0.0
        
        prd_face_mask_a   = np.expand_dims (prd_face_mask_a_0, axis=-1)
        prd_face_mask_aaa = np.repeat (prd_face_mask_a, (3,), axis=-1)

        img_prd_face_mask_aaa = cv2.warpAffine( prd_face_mask_aaa, face_mat, img_size, np.zeros(img_bgr.shape, dtype=float), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4 )
        img_prd_face_mask_aaa = np.clip (img_prd_face_mask_aaa, 0.0, 1.0)
            
        img_face_mask_aaa = img_prd_face_mask_aaa
        
        if debug:
            debugs += [img_face_mask_aaa.copy()]
        
        img_face_mask_aaa [ img_face_mask_aaa <= 0.1 ] = 0.0
            
        img_face_mask_flatten_aaa = img_face_mask_aaa.copy()
        img_face_mask_flatten_aaa[img_face_mask_flatten_aaa > 0.9] = 1.0

        maxregion = np.argwhere(img_face_mask_flatten_aaa==1.0)        

        out_img = img_bgr.copy()
        if maxregion.size != 0:
            miny,minx = maxregion.min(axis=0)[:2]
            maxy,maxx = maxregion.max(axis=0)[:2]
            lenx = maxx - minx
            leny = maxy - miny
            masky = int(minx+(lenx//2))
            maskx = int(miny+(leny//2))
            lowest_len = min (lenx, leny)
            
            if debug:
                print ("lowest_len = %f" % (lowest_len) )

            ero  = int( lowest_len * ( 0.126 - lowest_len * 0.00004551365 ) * 0.01*self.erode_mask_modifier )
            blur = int( lowest_len * 0.10                                   * 0.01*self.blur_mask_modifier )
          
            if debug:
                print ("ero = %d, blur = %d" % (ero, blur) )
                
            img_mask_blurry_aaa = img_face_mask_aaa
            if self.erode_mask:
                if ero > 0:
                    img_mask_blurry_aaa = cv2.erode(img_mask_blurry_aaa, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ero,ero)), iterations = 1 )
                elif ero < 0:
                    img_mask_blurry_aaa = cv2.dilate(img_mask_blurry_aaa, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(-ero,-ero)), iterations = 1 )

            if self.blur_mask and blur > 0:
                img_mask_blurry_aaa = cv2.blur(img_mask_blurry_aaa, (blur, blur) )
                
            img_mask_blurry_aaa = np.clip( img_mask_blurry_aaa, 0, 1.0 )
            
            if self.clip_border_mask_per > 0:
                prd_border_rect_mask_a = np.ones ( prd_face_mask_a.shape, dtype=prd_face_mask_a.dtype)        
                prd_border_size = int ( prd_border_rect_mask_a.shape[1] * self.clip_border_mask_per )

                prd_border_rect_mask_a[0:prd_border_size,:,:] = 0
                prd_border_rect_mask_a[-prd_border_size:,:,:] = 0
                prd_border_rect_mask_a[:,0:prd_border_size,:] = 0
                prd_border_rect_mask_a[:,-prd_border_size:,:] = 0
                prd_border_rect_mask_a = np.expand_dims(cv2.blur(prd_border_rect_mask_a, (prd_border_size, prd_border_size) ),-1)

            if self.mode == 'hist-match-bw':
                prd_face_bgr = cv2.cvtColor(prd_face_bgr, cv2.COLOR_BGR2GRAY)
                prd_face_bgr = np.repeat( np.expand_dims (prd_face_bgr, -1), (3,), -1 )
            
            if self.mode == 'hist-match' or self.mode == 'hist-match-bw':
                if debug:
                    debugs += [ cv2.warpAffine( prd_face_bgr, face_mat, img_size, np.zeros(img_bgr.shape, dtype=np.float32), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT ) ]
                    
                hist_mask_a = np.ones ( prd_face_bgr.shape[:2] + (1,) , dtype=prd_face_bgr.dtype)
                    
                if self.masked_hist_match:
                    hist_mask_a *= prd_face_mask_a

                new_prd_face_bgr = image_utils.color_hist_match(prd_face_bgr*hist_mask_a, dst_face_bgr*hist_mask_a )

                prd_face_bgr = new_prd_face_bgr
                    
            if self.mode == 'hist-match-bw':
                prd_face_bgr = prd_face_bgr.astype(np.float32)
                    
            out_img = cv2.warpAffine( prd_face_bgr, face_mat, img_size, out_img, cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT )

            if debug:
                debugs += [out_img.copy()]
                debugs += [img_mask_blurry_aaa.copy()]

            if self.mode == 'seamless' or self.mode == 'seamless-hist-match':
                out_img = np.clip( img_bgr*(1-img_face_mask_aaa) + (out_img*img_face_mask_aaa) , 0, 1.0 )
                if debug:
                    debugs += [out_img.copy()]
                out_img = cv2.seamlessClone( (out_img*255).astype(np.uint8), (img_bgr*255).astype(np.uint8), (img_face_mask_flatten_aaa*255).astype(np.uint8), (masky,maskx) , cv2.NORMAL_CLONE )
                out_img = out_img.astype(np.float32) / 255.0
                
                if debug:
                    debugs += [out_img.copy()]
                    
            if self.clip_border_mask_per > 0:
                img_prd_border_rect_mask_a = cv2.warpAffine( prd_border_rect_mask_a, face_mat, img_size, np.zeros(img_bgr.shape, dtype=np.float32), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT )
                img_prd_border_rect_mask_a = np.expand_dims (img_prd_border_rect_mask_a, -1)

                out_img = out_img * img_prd_border_rect_mask_a + img_bgr * (1.0 - img_prd_border_rect_mask_a)
                img_mask_blurry_aaa *= img_prd_border_rect_mask_a
            
            out_img =  np.clip( img_bgr*(1-img_mask_blurry_aaa) + (out_img*img_mask_blurry_aaa) , 0, 1.0 )

            if self.mode == 'seamless-hist-match':
                out_face_bgr = cv2.warpAffine( out_img, face_mat, (self.output_size, self.output_size) )                
                new_out_face_bgr = image_utils.color_hist_match(out_face_bgr, dst_face_bgr )                
                new_out = cv2.warpAffine( new_out_face_bgr, face_mat, img_size, img_bgr.copy(), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT )
                out_img =  np.clip( img_bgr*(1-img_mask_blurry_aaa) + (new_out*img_mask_blurry_aaa) , 0, 1.0 )
 
            if transfercolor:                                              #making transfer color from original DST image to fake
                from skimage import io, color
                lab_clr = color.rgb2lab(img_bgr)                            #original DST, converting RGB to LAB color space
                lab_bw = color.rgb2lab(out_img)                             #fake, converting RGB to LAB color space
                tmp_channel, a_channel, b_channel = cv2.split(lab_clr)      #taking color channel A and B from original dst image
                l_channel, tmp2_channel, tmp3_channel = cv2.split(lab_bw)   #taking lightness channel L from merged fake
                img_LAB = cv2.merge((l_channel,a_channel, b_channel))       #merging light and color
                out_img = color.lab2rgb(img_LAB)                            #converting LAB to RGB 
                
            if alpha:
                new_image = out_img.copy()
                new_image = (new_image*255).astype(np.uint8)                            #convert image to int
                b_channel, g_channel, r_channel = cv2.split(new_image)                  #splitting RGB
                alpha_channel = img_mask_blurry_aaa.copy()                              #making copy of alpha channel
                alpha_channel = (alpha_channel*255).astype(np.uint8)
                alpha_channel, tmp2, tmp3 = cv2.split(alpha_channel)                    #splitting alpha to three channels, they all same in original alpha channel, we need just one
                img_BGRA = cv2.merge((b_channel,g_channel, r_channel, alpha_channel))   #mergin RGB with alpha
                    
                out_img = img_BGRA
                out_img = out_img.astype(np.float32) / 255.0

        if debug:
            debugs += [out_img.copy()]
            
        return debugs if debug else out_img     
     
