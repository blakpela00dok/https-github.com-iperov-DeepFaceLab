import sys
import traceback
import queue
import threading
import time
from enum import Enum

import numpy as np
import itertools
from pathlib import Path
from utils import Path_utils
import imagelib
import cv2
import models
from interact import interact as io

def trainerThread (s2c, c2s, e, args, device_args):
    while True:
        try:
            start_time = time.time()

            training_data_src_path = Path( args.get('training_data_src_dir', '') )
            training_data_dst_path = Path( args.get('training_data_dst_dir', '') )

            pretraining_data_path = args.get('pretraining_data_dir', '')
            pretraining_data_path = Path(pretraining_data_path) if pretraining_data_path is not None else None

            model_path = Path( args.get('model_path', '') )
            model_name = args.get('model_name', '')
            save_interval_min = 15
            debug = args.get('debug', '')
            execute_programs = args.get('execute_programs', [])

            if not training_data_src_path.exists():
                io.log_err('Training data src directory does not exist.')
                break

            if not training_data_dst_path.exists():
                io.log_err('Training data dst directory does not exist.')
                break

            if not model_path.exists():
                model_path.mkdir(exist_ok=True)

            model = models.import_model(model_name)(
                        model_path,
                        training_data_src_path=training_data_src_path,
                        training_data_dst_path=training_data_dst_path,
                        pretraining_data_path=pretraining_data_path,
                        debug=debug,
                        device_args=device_args)

            is_reached_goal = model.is_reached_iter_goal()

            shared_state = { 'after_save' : False }
            loss_string = ""
            save_iter =  model.get_iter()
            def model_save():
                if not debug and not is_reached_goal:
                    io.log_info ("Saving....", end='\r')
                    model.save()
                    shared_state['after_save'] = True

            def send_preview():
                if not debug:
                    previews = model.get_previews()
                    c2s.put ( {'op':'show', 'previews': previews, 'iter':model.get_iter(), 'loss_history': model.get_loss_history().copy() } )
                else:
                    previews = [( 'debug, press update for new', model.debug_one_iter())]
                    c2s.put ( {'op':'show', 'previews': previews} )
                e.set() #Set the GUI Thread as Ready


            if model.is_first_run():
                model_save()

            if model.get_target_iter() != 0:
                if is_reached_goal:
                    io.log_info('Model already trained to target iteration. You can use preview.')
                else:
                    io.log_info('Starting. Target iteration: %d. Press "Enter" to stop training and save model.' % ( model.get_target_iter()  ) )
            else:
                io.log_info('Starting. Press "Enter" to stop training and save model.')

            last_save_time = time.time()

            execute_programs = [ [x[0], x[1], time.time() ] for x in execute_programs ]

            for i in itertools.count(0,1):
                if not debug:
                    cur_time = time.time()

                    for x in execute_programs:
                        prog_time, prog, last_time = x
                        exec_prog = False
                        if prog_time > 0 and (cur_time - start_time) >= prog_time:
                            x[0] = 0
                            exec_prog = True
                        elif prog_time < 0 and (cur_time - last_time)  >= -prog_time:
                            x[2] = cur_time
                            exec_prog = True

                        if exec_prog:
                            try:
                                exec(prog)
                            except Exception as e:
                                print("Unable to execute program: %s" % (prog) )

                    if not is_reached_goal:
                        iter, iter_time, batch_size = model.train_one_iter()

                        loss_history = model.get_loss_history()
                        time_str = time.strftime("[%H:%M:%S]")
                        if iter_time >= 10:
                            loss_string = "{0}[#{1:06d}][{2:.5s}s][bs: {3}]".format ( time_str, iter, '{:0.4f}'.format(iter_time), batch_size )
                        else:
                            loss_string = "{0}[#{1:06d}][{2:04d}ms][bs: {3}]".format ( time_str, iter, int(iter_time*1000), batch_size)

                        if shared_state['after_save']:
                            shared_state['after_save'] = False
                            last_save_time = time.time() #upd last_save_time only after save+one_iter, because plaidML rebuilds programs after save https://github.com/plaidml/plaidml/issues/274

                            mean_loss = np.mean ( [ np.array(loss_history[i]) for i in range(save_iter, iter) ], axis=0)

                            for loss_value in mean_loss:
                                loss_string += "[%.4f]" % (loss_value)

                            io.log_info (loss_string)

                            save_iter = iter
                        else:
                            for loss_value in loss_history[-1]:
                                loss_string += "[%.4f]" % (loss_value)

                            if io.is_colab():
                                io.log_info ('\r' + loss_string, end='')
                            else:
                                io.log_info (loss_string, end='\r')

                        if model.get_target_iter() != 0 and model.is_reached_iter_goal():
                            io.log_info ('Reached target iteration.')
                            model_save()
                            is_reached_goal = True
                            io.log_info ('You can use preview now.')

                if not is_reached_goal and (time.time() - last_save_time) >= save_interval_min*60:
                    model_save()
                    send_preview()

                if i==0:
                    if is_reached_goal:
                        model.pass_one_iter()
                    send_preview()

                if debug:
                    time.sleep(0.005)

                while not s2c.empty():
                    input = s2c.get()
                    op = input['op']
                    if op == 'save':
                        model_save()
                    elif op == 'preview':
                        if is_reached_goal:
                            model.pass_one_iter()
                        send_preview()
                    elif op == 'close':
                        model_save()
                        i = -1
                        break

                if i == -1:
                    break



            model.finalize()

        except Exception as e:
            print ('Error: %s' % (str(e)))
            traceback.print_exc()
        break
    c2s.put ( {'op':'close'} )


class Zoom(Enum):
    ZOOM_25 = (1/4, '25%')
    ZOOM_33 = (1/3, '33%')
    ZOOM_50 = (1/2, '50%')
    ZOOM_67 = (2/3, '67%')
    ZOOM_75 = (3/4, '75%')
    ZOOM_80 = (4/5, '80%')
    ZOOM_90 = (9/10, '90%')
    ZOOM_100 = (1, '100%')
    ZOOM_110 = (11/10, '110%')
    ZOOM_125 = (5/4, '125%')
    ZOOM_150 = (3/2, '150%')
    ZOOM_175 = (7/4, '175%')
    ZOOM_200 = (2, '200%')
    ZOOM_250 = (5/2, '250%')
    ZOOM_300 = (3, '300%')
    ZOOM_400 = (4, '400%')
    ZOOM_500 = (5, '500%')

    def __init__(self, scale, label):
        self.scale = scale
        self.label = label

    def prev(self):
        cls = self.__class__
        members = list(cls)
        index = members.index(self) - 1
        if index < 0:
            return self
        return members[index]

    def next(self):
        cls = self.__class__
        members = list(cls)
        index = members.index(self) + 1
        if index >= len(members):
            return self
        return members[index]


def scale_previews(previews, zoom=Zoom.ZOOM_100):
    scaled = []
    for preview in previews:
        preview_name, preview_rgb = preview
        scale_factor = zoom.scale
        if scale_factor < 1:
            scaled.append((preview_name, cv2.resize(preview_rgb, (0, 0),
                                                    fx=scale_factor,
                                                    fy=scale_factor,
                                                    interpolation=cv2.INTER_AREA)))
        elif scale_factor > 1:
            scaled.append((preview_name, cv2.resize(preview_rgb, (0, 0),
                                                    fx=scale_factor,
                                                    fy=scale_factor,
                                                    interpolation=cv2.INTER_LANCZOS4)))
        else:
            scaled.append((preview_name, preview_rgb))
    return scaled


def create_preview_pane_image(previews, selected_preview, loss_history,
                              show_last_history_iters_count, iteration, batch_size, zoom=Zoom.ZOOM_100):
    selected_preview_name = previews[selected_preview][0]
    selected_preview_rgb = previews[selected_preview][1]
    h, w, c = selected_preview_rgb.shape

    # HEAD
    head_lines = [
        '[s]:save [enter]:exit [-/+]:zoom: %s' % zoom.label,
        '[p]:update [space]:next preview [l]:change history range',
        'Preview: "%s" [%d/%d]' % (selected_preview_name,selected_preview+1, len(previews))
    ]
    head_line_height = int(15 * zoom.scale)
    head_height = len(head_lines) * head_line_height
    head = np.ones((head_height, w, c)) * 0.1

    for i in range(0, len(head_lines)):
        t = i * head_line_height
        b = (i+1) * head_line_height
        head[t:b, 0:w] += imagelib.get_text_image((head_line_height, w, c), head_lines[i], color=[0.8]*c)

    final = head

    if loss_history is not None:
        if show_last_history_iters_count == 0:
            loss_history_to_show = loss_history
        else:
            loss_history_to_show = loss_history[-show_last_history_iters_count:]
        lh_height = int(100 * zoom.scale)
        lh_img = models.ModelBase.get_loss_history_preview(loss_history_to_show, iteration, batch_size, w, c, lh_height)
        final = np.concatenate ( [final, lh_img], axis=0 )

    final = np.concatenate([final, selected_preview_rgb], axis=0)
    final = np.clip(final, 0, 1)
    return (final*255).astype(np.uint8)


def main(args, device_args):
    io.log_info ("Running trainer.\r\n")

    no_preview = args.get('no_preview', False)


    s2c = queue.Queue()
    c2s = queue.Queue()

    e = threading.Event()
    thread = threading.Thread(target=trainerThread, args=(s2c, c2s, e, args, device_args) )
    thread.start()

    e.wait() #Wait for inital load to occur.

    if no_preview:
        while True:
            if not c2s.empty():
                input = c2s.get()
                op = input.get('op','')
                if op == 'close':
                    break
            try:
                io.process_messages(0.1)
            except KeyboardInterrupt:
                s2c.put ( {'op': 'close'} )
    else:
        wnd_name = "Training preview"
        io.named_window(wnd_name)
        io.capture_keys(wnd_name)

        previews = None
        loss_history = None
        selected_preview = 0
        update_preview = False
        is_showing = False
        is_waiting_preview = False
        show_last_history_iters_count = 0
        iteration = 0
        batch_size = 1
        zoom = Zoom.ZOOM_100

        while True:
            if not c2s.empty():
                input = c2s.get()
                op = input['op']
                if op == 'show':
                    is_waiting_preview = False
                    loss_history = input['loss_history'] if 'loss_history' in input.keys() else None
                    previews = input['previews'] if 'previews' in input.keys() else None
                    iteration = input['iter'] if 'iter' in input.keys() else 0
                    #batch_size = input['batch_size'] if 'iter' in input.keys() else 1
                    if previews is not None:
                        previews = scale_previews(previews, zoom)
                        update_preview = True
                elif op == 'close':
                    break

            if update_preview:
                update_preview = False
                selected_preview = selected_preview % len(previews)
                preview_pane_image = create_preview_pane_image(previews,
                                                               selected_preview,
                                                               loss_history,
                                                               show_last_history_iters_count,
                                                               iteration,
                                                               batch_size,
                                                               zoom)
                io.show_image(wnd_name, preview_pane_image)
                is_showing = True

            key_events = io.get_key_events(wnd_name)
            key, chr_key, ctrl_pressed, alt_pressed, shift_pressed = key_events[-1] if len(key_events) > 0 else (0,0,False,False,False)

            if key == ord('\n') or key == ord('\r'):
                s2c.put ( {'op': 'close'} )
            elif key == ord('s'):
                s2c.put ( {'op': 'save'} )
            elif key == ord('p'):
                if not is_waiting_preview:
                    is_waiting_preview = True
                    s2c.put ( {'op': 'preview'} )
            elif key == ord('l'):
                if show_last_history_iters_count == 0:
                    show_last_history_iters_count = 5000
                elif show_last_history_iters_count == 5000:
                    show_last_history_iters_count = 10000
                elif show_last_history_iters_count == 10000:
                    show_last_history_iters_count = 50000
                elif show_last_history_iters_count == 50000:
                    show_last_history_iters_count = 100000
                elif show_last_history_iters_count == 100000:
                    show_last_history_iters_count = 0
                update_preview = True
            elif key == ord(' '):
                selected_preview = (selected_preview + 1) % len(previews)
                update_preview = True
            elif key == ord('-'):
                zoom = zoom.prev()
                update_preview = True
            elif key == ord('=') or key == ord('+'):
                zoom = zoom.next()
                update_preview = True
            try:
                io.process_messages(0.1)
            except KeyboardInterrupt:
                s2c.put ( {'op': 'close'} )

        io.destroy_all_windows()
