import traceback
from pathlib import Path

import imagelib
from interact import interact as io
from nnlib import DeepPortraitRelighting
from utils import Path_utils
from utils.cv2_utils import *
from DFLIMG import *

class RelightEditor:
    def __init__(self, image_paths, dpr, lighten):
        self.image_paths = image_paths
        self.dpr = dpr
        self.lighten = lighten

        self.current_img_path = None
        self.current_img = None
        self.current_img_shape = None
        self.pick_new_face()

        self.alt_azi_ar = [ [0,0,1.0] ]
        self.alt_azi_cur = 0

        self.mouse_x = self.mouse_y = 9999
        self.screen_status_block = None
        self.screen_status_block_dirty = True
        self.screen_changed = True

    def pick_new_face(self):
        self.current_img_path = self.image_paths[ np.random.randint(len(self.image_paths)) ]
        self.current_img = cv2_imread (str(self.current_img_path))
        self.current_img_shape = self.current_img.shape
        self.set_screen_changed()

    def set_screen_changed(self):
        self.screen_changed = True

    def switch_screen_changed(self):
        result = self.screen_changed
        self.screen_changed = False
        return result

    def make_screen(self):
        alt,azi,inten=self.alt_azi_ar[self.alt_azi_cur]

        img = self.dpr.relight (self.current_img, alt, azi, inten, self.lighten)

        h,w,c = img.shape

        lines = ['Pick light directions for whole faceset.',
                 '[q]-new test face',
                 '[w][e]-navigate',
                 '[a][s]-intensity',
                 '[r]-new [t]-delete [enter]-process',
                 '']

        for i, (alt,azi,inten) in enumerate(self.alt_azi_ar):
            s = '>:' if self.alt_azi_cur == i else ' :'
            s += f'alt=[{ int(alt):03}] azi=[{ int(azi):03}] int=[{inten:01.1f}]'
            lines += [ s ]

        lines_count = len(lines)
        h_line = 16

        sh = lines_count * h_line
        sw = 400
        sc = c
        status_img = np.ones ( (sh,sw,sc) ) * 0.1

        for i in range(lines_count):
            status_img[ i*h_line:(i+1)*h_line, 0:sw] += \
                imagelib.get_text_image (  (h_line,sw,c), lines[i], color=[0.8]*c )

        status_img = np.clip(status_img*255, 0, 255).astype(np.uint8)

        #combine screens
        if sh > h:
            img = np.concatenate ([img, np.zeros( (sh-h,w,c), dtype=img.dtype ) ], axis=0)
        elif h > sh:
            status_img = np.concatenate ([status_img, np.zeros( (h-sh,sw,sc), dtype=img.dtype ) ], axis=0)

        img = np.concatenate ([img, status_img], axis=1)

        return img

    def run(self):
        wnd_name = "Relighter"
        io.named_window(wnd_name)
        io.capture_keys(wnd_name)
        io.capture_mouse(wnd_name)

        zoom_factor = 1.0

        is_angle_editing = False

        is_exit = False
        while not is_exit:
            io.process_messages(0.0001)

            mouse_events = io.get_mouse_events(wnd_name)
            for ev in mouse_events:
                (x, y, ev, flags) = ev
                if ev == io.EVENT_LBUTTONDOWN:
                    is_angle_editing = True

                if ev == io.EVENT_LBUTTONUP:
                    is_angle_editing = False

                if is_angle_editing:
                    h,w,c = self.current_img_shape
                    
                    alt,azi,inten = self.alt_azi_ar[self.alt_azi_cur]
                    alt = np.clip ( ( 0.5-y/w )*2.0,   -1, 1)*90
                    azi = np.clip ( (x / h - 0.5)*2.0, -1, 1)*90
                    self.alt_azi_ar[self.alt_azi_cur] = (alt,azi,inten)
                        

                    self.set_screen_changed()

            key_events = io.get_key_events(wnd_name)
            key, chr_key, ctrl_pressed, alt_pressed, shift_pressed = key_events[-1] if len(key_events) > 0 else (0,0,False,False,False)

            if key != 0:
                if chr_key == 'q':
                    self.pick_new_face()
                elif chr_key == 'w':
                    self.alt_azi_cur = np.clip (self.alt_azi_cur-1, 0, len(self.alt_azi_ar)-1)
                    self.set_screen_changed()
                elif chr_key == 'e':
                    self.alt_azi_cur = np.clip (self.alt_azi_cur+1, 0, len(self.alt_azi_ar)-1)
                    self.set_screen_changed()
                elif chr_key == 'r':
                    #add direction
                    self.alt_azi_ar += [ [0,0,1.0] ]
                    self.alt_azi_cur +=1
                    self.set_screen_changed()
                elif chr_key == 't':
                    if len(self.alt_azi_ar) > 1:
                        self.alt_azi_ar.pop(self.alt_azi_cur)
                        self.alt_azi_cur = np.clip (self.alt_azi_cur, 0, len(self.alt_azi_ar)-1)
                        self.set_screen_changed()
                elif chr_key == 'a':
                    alt,azi,inten = self.alt_azi_ar[self.alt_azi_cur]
                    inten = np.clip ( inten-0.1, 0.0, 1.0)
                    self.alt_azi_ar[self.alt_azi_cur] = (alt,azi,inten)
                    self.set_screen_changed()
                elif chr_key == 's':
                    alt,azi,inten = self.alt_azi_ar[self.alt_azi_cur]
                    inten = np.clip ( inten+0.1, 0.0, 1.0)
                    self.alt_azi_ar[self.alt_azi_cur] = (alt,azi,inten)
                    self.set_screen_changed()
                elif key == 27 or chr_key == '\r' or chr_key == '\n': #esc
                    is_exit = True

            if self.switch_screen_changed():
                screen = self.make_screen()
                if zoom_factor != 1.0:
                    h,w,c = screen.shape
                    screen = cv2.resize ( screen, ( int(w*zoom_factor), int(h*zoom_factor) ) )
                io.show_image (wnd_name, screen )

        io.destroy_window(wnd_name)

        return self.alt_azi_ar

def relight(input_dir, lighten=None, random_one=None):
    if lighten is None:
        lighten = io.input_bool ("Lighten the faces? ( y/n default:n ?:help ) : ", False, help_message="Lighten the faces instead of shadow. May produce artifacts." )

    if io.is_colab():
        io.log_info("In colab version you cannot choose light directions manually.")
        manual = False
    else:
        manual = io.input_bool ("Choose light directions manually? ( y/n default:y ) : ", True)

    if not manual:
        if random_one is None:
            random_one = io.input_bool ("Relight the faces only with one random direction and random intensity? ( y/n default:y ?:help) : ", True, help_message="Otherwise faceset will be relighted with predefined 7 light directions but with random intensity.")

    image_paths = [Path(x) for x in Path_utils.get_image_paths(input_dir)]
    filtered_image_paths = []
    for filepath in io.progress_bar_generator(image_paths, "Collecting fileinfo"):
        try:
            dflimg = DFLIMG.load (Path(filepath))

            if dflimg is None:
                io.log_err ("%s is not a dfl image file" % (filepath.name) )
            else:
                if not dflimg.get_relighted():
                    filtered_image_paths += [filepath]
        except:
            io.log_err (f"Exception occured while processing file {filepath.name}. Error: {traceback.format_exc()}")
    image_paths = filtered_image_paths

    if len(image_paths) == 0:
        io.log_info("No files to process.")
        return

    dpr = DeepPortraitRelighting()

    if manual:
        alt_azi_ar = RelightEditor(image_paths, dpr, lighten).run()
        
    for filepath in io.progress_bar_generator(image_paths, "Relighting"):
        try:
            dflimg = DFLIMG.load ( Path(filepath) )
            if dflimg is None:
                io.log_err ("%s is not a dfl image file" % (filepath.name) )
                continue
            else:
                if dflimg.get_relighted():
                    continue
                img = cv2_imread (str(filepath))

                if random_one:
                    alt = np.random.randint(-90,91)
                    azi = np.random.randint(-90,91)
                    inten = np.random.random()*0.3+0.3
                    relighted_imgs = [dpr.relight(img,alt=alt,azi=azi,intensity=inten,lighten=lighten)]
                else:
                    if not manual and not random_one:
                        inten = np.random.random()*0.3+0.3
                        alt_azi_ar = [(60,0,inten), (60,60,inten), (0,60,inten), (-60,60,inten), (-60,0,inten), (-60,-60,inten), (0,-60,inten), (60,-60,inten)]
                    
                    relighted_imgs = [dpr.relight(img,alt=alt,azi=azi,intensity=inten,lighten=lighten) for (alt,azi,inten) in alt_azi_ar ]

                i = 0
                for i,relighted_img in enumerate(relighted_imgs):
                    im_flags = []
                    if filepath.suffix == '.jpg':
                        im_flags += [int(cv2.IMWRITE_JPEG_QUALITY), 100]

                    while True:
                        relighted_filepath = filepath.parent / (filepath.stem+f'_relighted_{i}'+filepath.suffix)
                        if not relighted_filepath.exists():
                            break
                        i += 1

                    cv2_imwrite (relighted_filepath, relighted_img )
                    
                    dflimg.remove_source_filename()
                    dflimg.embed_and_set (relighted_filepath, relighted=True )
        except:
            io.log_err (f"Exception occured while processing file {filepath.name}. Error: {traceback.format_exc()}")

def delete_relighted(input_dir):
    input_path = Path(input_dir)
    image_paths = [Path(x) for x in Path_utils.get_image_paths(input_path)]

    files_to_delete = []
    for filepath in io.progress_bar_generator(image_paths, "Loading"):
        dflimg = DFLIMG.load ( Path(filepath) )

        if dflimg is None:
            io.log_err ("%s is not a dfl image file" % (filepath.name) )
            continue
        else:
            if dflimg.get_relighted():
                files_to_delete += [filepath]

    for file in io.progress_bar_generator(files_to_delete, "Deleting"):
        file.unlink()
