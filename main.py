import os
import sys
import argparse
from utils import Path_utils
from utils import os_utils
from pathlib import Path
import numpy as np

if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 2):
    raise Exception("This program requires at least Python 3.2")

class fixPathAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    os_utils.set_process_lowest_prio()
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf-suppress-std', action="store_true", dest="tf_suppress_std", default=False, help="Suppress tensorflow initialization info. May not works on some python builds such as anaconda python 3.6.4. If you can fix it, you are welcome.")

    subparsers = parser.add_subparsers()

    def process_extract(arguments):
        from mainscripts import Extractor
        Extractor.main (
            input_dir=arguments.input_dir,
            output_dir=arguments.output_dir,
            debug=arguments.debug,
            face_type=arguments.face_type,
            detector=arguments.detector,
            multi_gpu=arguments.multi_gpu,
            cpu_only=arguments.cpu_only,
            manual_fix=arguments.manual_fix,
            manual_window_size=arguments.manual_window_size
            )

    extract_parser = subparsers.add_parser( "extract", help="Extract the faces from a pictures.")
    extract_parser.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    extract_parser.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir", help="Output directory. This is where the extracted files will be stored.")
    extract_parser.add_argument('--debug', action="store_true", dest="debug", default=False, help="Writes debug images to [output_dir]_debug\ directory.")
    extract_parser.add_argument('--face-type', dest="face_type", choices=['half_face', 'full_face', 'head', 'avatar', 'mark_only'], default='full_face', help="Default 'full_face'. Don't change this option, currently all models uses 'full_face'")
    extract_parser.add_argument('--detector', dest="detector", choices=['dlib','mt','manual'], default='dlib', help="Type of detector. Default 'dlib'. 'mt' (MTCNNv1) - faster, better, almost no jitter, perfect for gathering thousands faces for src-set. It is also good for dst-set, but can generate false faces in frames where main face not recognized! In this case for dst-set use either 'dlib' with '--manual-fix' or '--detector manual'. Manual detector suitable only for dst-set.")
    extract_parser.add_argument('--multi-gpu', action="store_true", dest="multi_gpu", default=False, help="Enables multi GPU.")
    extract_parser.add_argument('--manual-fix', action="store_true", dest="manual_fix", default=False, help="Enables manual extract only frames where faces were not recognized.")
    extract_parser.add_argument('--manual-window-size', type=int, dest="manual_window_size", default=0, help="Manual fix window size. Example: 1368. Default: frame size.")
    extract_parser.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Extract on CPU. Forces to use MT extractor.")


    extract_parser.set_defaults (func=process_extract)

    def process_sort(arguments):
        from mainscripts import Sorter
        Sorter.main (input_path=arguments.input_dir, sort_by_method=arguments.sort_by_method)

    sort_parser = subparsers.add_parser( "sort", help="Sort faces in a directory.")
    sort_parser.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    sort_parser.add_argument('--by', required=True, dest="sort_by_method", choices=("blur", "face", "face-dissim", "face-yaw", "hist", "hist-dissim", "brightness", "hue", "black", "origname"), help="Method of sorting. 'origname' sort by original filename to recover original sequence." )
    sort_parser.set_defaults (func=process_sort)

    def process_train(arguments):
        if 'DFL_TARGET_EPOCH' in os.environ.keys():
            arguments.target_epoch = int ( os.environ['DFL_TARGET_EPOCH'] )

        if 'DFL_BATCH_SIZE' in os.environ.keys():
            arguments.batch_size = int ( os.environ['DFL_BATCH_SIZE'] )
        from mainscripts import Trainer
        Trainer.main (
            training_data_src_dir=arguments.training_data_src_dir,
            training_data_dst_dir=arguments.training_data_dst_dir,
            model_path=arguments.model_dir,
            model_name=arguments.model_name,
            debug              = arguments.debug,
            preview           = arguments.preview,
            #**options
            batch_size         = arguments.batch_size,
            write_preview_history = arguments.write_preview_history,
            target_epoch       = arguments.target_epoch,
            save_interval_min  = arguments.save_interval_min,
            choose_worst_gpu   = arguments.choose_worst_gpu,
            force_best_gpu_idx = arguments.force_best_gpu_idx,
            multi_gpu          = arguments.multi_gpu,
            force_gpu_idxs     = arguments.force_gpu_idxs,
            cpu_only           = arguments.cpu_only
            )

    train_parser = subparsers.add_parser( "train", help="Trainer")
    train_parser.add_argument('--training-data-src-dir', required=True, action=fixPathAction, dest="training_data_src_dir", help="Dir of src-set.")
    train_parser.add_argument('--training-data-dst-dir', required=True, action=fixPathAction, dest="training_data_dst_dir", help="Dir of dst-set.")
    train_parser.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir", help="Model dir.")
    train_parser.add_argument('--model', required=True, dest="model_name", choices=Path_utils.get_all_dir_names_startswith ( Path(__file__).parent / 'models' , 'Model_'), help="Type of model")
    train_parser.add_argument('--write-preview-history', action="store_true", dest="write_preview_history", default=False, help="Enable write preview history.")
    train_parser.add_argument('--debug', action="store_true", dest="debug", default=False, help="Debug training.")
    train_parser.add_argument('--batch-size', type=int, dest="batch_size", default=0, help="Model batch size. Default - auto. Environment variable: ODFS_BATCH_SIZE.")
    train_parser.add_argument('--target-epoch', type=int, dest="target_epoch", default=0, help="Train until target epoch. Default - unlimited. Environment variable: ODFS_TARGET_EPOCH.")
    train_parser.add_argument('--save-interval-min', type=int, dest="save_interval_min", default=10, help="Save interval in minutes. Default 10.")
    train_parser.add_argument('--choose-worst-gpu', action="store_true", dest="choose_worst_gpu", default=False, help="Choose worst GPU instead of best.")
    train_parser.add_argument('--force-best-gpu-idx', type=int, dest="force_best_gpu_idx", default=-1, help="Force to choose this GPU idx as best(worst).")
    train_parser.add_argument('--multi-gpu', action="store_true", dest="multi_gpu", default=False, help="MultiGPU option. It will select only same best(worst) GPU models.")
    train_parser.add_argument('--force-gpu-idxs', type=str, dest="force_gpu_idxs", default=None, help="Override final GPU idxs. Example: 0,1,2.")
    train_parser.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Train on CPU.")
    train_parser.add_argument('--preview', action="store_true",dest="preview", default=False, help="Show preview.")

    train_parser.set_defaults (func=process_train)

    def process_convert(arguments):
        if arguments.ask_for_params:
            try:
                mode = int ( input ("Choose mode: (1) hist match, (2) hist match bw, (3) seamless (default), (4) seamless hist match : ") )
            except:
                mode = 3

            if mode == 1:
                arguments.mode = 'hist-match'
            elif mode == 2:
                arguments.mode = 'hist-match-bw'
            elif mode == 3:
                arguments.mode = 'seamless'
            elif mode == 4:
                arguments.mode = 'seamless-hist-match'

            if arguments.mode == 'hist-match' or arguments.mode == 'hist-match-bw':
                try:
                    arguments.masked_hist_match = bool ( {"1":True,"0":False}[input("Masked hist match? [0 or 1] (default 1) : ").lower()] )
                except:
                    arguments.masked_hist_match = True

            if arguments.mode == 'hist-match' or arguments.mode == 'hist-match-bw' or arguments.mode == 'seamless-hist-match':
                try:
                    hist_match_threshold = int ( input ("Hist match threshold. [0..255] (default - 255) : ") )
                    arguments.hist_match_threshold = hist_match_threshold
                except:
                    arguments.hist_match_threshold = 255

            try:
                arguments.use_predicted_mask = bool ( {"1":True,"0":False}[input("Use predicted mask? [0 or 1] (default 1) : ").lower()] )
            except:
                arguments.use_predicted_mask = False

            try:
                arguments.erode_mask_modifier = int ( input ("Choose erode mask modifier [-200..200] (default 0) : ") )
            except:
                arguments.erode_mask_modifier = 0

            try:
                arguments.blur_mask_modifier = int ( input ("Choose blur mask modifier [-200..200] (default 0) : ") )
            except:
                arguments.blur_mask_modifier = 0

            if arguments.mode == 'seamless' or arguments.mode == 'seamless-hist-match':
                try:
                    arguments.seamless_erode_mask_modifier = int ( input ("Choose seamless erode mask modifier [-100..100] (default 0) : ") )
                except:
                    arguments.seamless_erode_mask_modifier = 0

            try:
                arguments.output_face_scale_modifier = int ( input ("Choose output face scale modifier [-50..50] (default 0) : ") )
            except:
                arguments.output_face_scale_modifier = 0

            try:
                arguments.transfercolor = bool ( {"1":True,"0":False}[input("Transfer color from dst face to converted final face? [0 or 1] (default 0) : ").lower()] )
            except:
                arguments.transfercolor = False

            try:
                arguments.final_image_color_degrade_power = int ( input ("Degrade color power of final image [0..100] (default 0) : ") )
            except:
                arguments.final_image_color_degrade_power = 0

            try:
                arguments.alpha = bool ( {"1":True,"0":False}[input("Export png with alpha channel? [0..1] (default 0) : ").lower()] )
            except:
                arguments.alpha = False

        arguments.erode_mask_modifier = np.clip ( int(arguments.erode_mask_modifier), -200, 200)
        arguments.blur_mask_modifier = np.clip ( int(arguments.blur_mask_modifier), -200, 200)
        arguments.seamless_erode_mask_modifier = np.clip ( int(arguments.seamless_erode_mask_modifier), -100, 100)
        arguments.output_face_scale_modifier = np.clip ( int(arguments.output_face_scale_modifier), -50, 50)

        from mainscripts import Converter
        Converter.main (
            input_dir=arguments.input_dir,
            output_dir=arguments.output_dir,
            aligned_dir=arguments.aligned_dir,
            model_dir=arguments.model_dir,
            model_name=arguments.model_name,
            debug = arguments.debug,
            mode = arguments.mode,
            masked_hist_match = arguments.masked_hist_match,
            hist_match_threshold = arguments.hist_match_threshold,
            use_predicted_mask = arguments.use_predicted_mask,
            erode_mask_modifier = arguments.erode_mask_modifier,
            blur_mask_modifier = arguments.blur_mask_modifier,
            seamless_erode_mask_modifier = arguments.seamless_erode_mask_modifier,
            output_face_scale_modifier = arguments.output_face_scale_modifier,
            final_image_color_degrade_power = arguments.final_image_color_degrade_power,
            transfercolor = arguments.transfercolor,
            alpha = arguments.alpha,
            force_best_gpu_idx = arguments.force_best_gpu_idx,
            cpu_only = arguments.cpu_only
            )

    convert_parser = subparsers.add_parser( "convert", help="Converter")
    convert_parser.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    convert_parser.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir", help="Output directory. This is where the converted files will be stored.")
    convert_parser.add_argument('--aligned-dir', action=fixPathAction, dest="aligned_dir", help="Aligned directory. This is where the extracted of dst faces stored. Not used in AVATAR model.")
    convert_parser.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir", help="Model dir.")
    convert_parser.add_argument('--model', required=True, dest="model_name", choices=Path_utils.get_all_dir_names_startswith ( Path(__file__).parent / 'models' , 'Model_'), help="Type of model")
    convert_parser.add_argument('--ask-for-params', action="store_true", dest="ask_for_params", default=False, help="Ask for params.")
    convert_parser.add_argument('--mode',  dest="mode", choices=['seamless','hist-match', 'hist-match-bw','seamless-hist-match'], default='seamless', help="Face overlaying mode. Seriously affects result.")
    convert_parser.add_argument('--masked-hist-match', type=str2bool, nargs='?', const=True, default=True, help="True or False. Excludes background for hist match. Default - True.")
    convert_parser.add_argument('--hist-match-threshold', type=int, dest="hist_match_threshold", default=255, help="Hist match threshold. Decrease to hide artifacts of hist match. Valid range [0..255]. Default 255")
    convert_parser.add_argument('--use-predicted-mask', action="store_true", dest="use_predicted_mask", default=True, help="Use predicted mask by model. Default - True.")
    convert_parser.add_argument('--erode-mask-modifier', type=int, dest="erode_mask_modifier", default=0, help="Automatic erode mask modifier. Valid range [-200..200].")
    convert_parser.add_argument('--blur-mask-modifier', type=int, dest="blur_mask_modifier", default=0, help="Automatic blur mask modifier. Valid range [-200..200].")
    convert_parser.add_argument('--seamless-erode-mask-modifier', type=int, dest="seamless_erode_mask_modifier", default=0, help="Automatic seamless erode mask modifier. Valid range [-200..200].")
    convert_parser.add_argument('--output-face-scale-modifier', type=int, dest="output_face_scale_modifier", default=0, help="Output face scale modifier. Valid range [-50..50].")
    convert_parser.add_argument('--final-image-color-degrade-power', type=int, dest="final_image_color_degrade_power", default=0, help="Degrades colors of final image to hide face problems. Valid range [0..100].")
    convert_parser.add_argument('--transfercolor', action="store_true", dest="transfercolor", default=False, help="Transfer color from dst face to converted final face.")
    convert_parser.add_argument('--alpha', action="store_true", dest="alpha", default=False, help="Embeds alpha channel of face mask to final PNG. Used in manual composing video by editors such as Sony Vegas or After Effects.")
    convert_parser.add_argument('--debug', action="store_true", dest="debug", default=False, help="Debug converter.")
    convert_parser.add_argument('--force-best-gpu-idx', type=int, dest="force_best_gpu_idx", default=-1, help="Force to choose this GPU idx as best.")
    convert_parser.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Convert on CPU.")

    convert_parser.set_defaults(func=process_convert)

    def bad_args(arguments):
        parser.print_help()
        exit(0)
    parser.set_defaults(func=bad_args)

    arguments = parser.parse_args()
    if arguments.tf_suppress_std:
        os.environ['TF_SUPPRESS_STD'] = '1'

    arguments.func(arguments)

    print ("Done.")
'''
import code
code.interact(local=dict(globals(), **locals()))
'''
