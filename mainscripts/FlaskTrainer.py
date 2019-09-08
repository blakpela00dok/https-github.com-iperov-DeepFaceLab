import sys
import traceback
import queue
import threading
import time
from io import BytesIO
import base64
import numpy as np
import itertools
from pathlib import Path
from utils import Path_utils
import imagelib
import cv2
import models
from interact import interact as io
from flask import Flask, send_file, Response, render_template, render_template_string, request, g
from flask_caching import Cache

def trainerThread(s2c, c2s, e, args, device_args):
    while True:
        try:
            start_time = time.time()

            training_data_src_path = Path(args.get('training_data_src_dir', ''))
            training_data_dst_path = Path(args.get('training_data_dst_dir', ''))

            pretraining_data_path = args.get('pretraining_data_dir', '')
            pretraining_data_path = Path(pretraining_data_path) if pretraining_data_path is not None else None

            model_path = Path(args.get('model_path', ''))
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

            shared_state = {'after_save': False}
            loss_string = ""
            save_iter = model.get_iter()

            def model_save():
                if not debug and not is_reached_goal:
                    io.log_info("Saving....", end='\r')
                    model.save()
                    shared_state['after_save'] = True

            def send_preview():
                if not debug:
                    previews = model.get_previews()
                    c2s.put({'op': 'show', 'previews': previews, 'iter': model.get_iter(),
                             'loss_history': model.get_loss_history().copy()})
                else:
                    previews = [('debug, press update for new', model.debug_one_iter())]
                    c2s.put({'op': 'show', 'previews': previews})
                e.set()  # Set the GUI Thread as Ready

            if model.is_first_run():
                model_save()

            if model.get_target_iter() != 0:
                if is_reached_goal:
                    io.log_info('Model already trained to target iteration. You can use preview.')
                else:
                    io.log_info('Starting. Target iteration: %d. Press "Enter" to stop training and save model.' % (
                        model.get_target_iter()))
            else:
                io.log_info('Starting. Press "Enter" to stop training and save model.')

            last_save_time = time.time()

            execute_programs = [[x[0], x[1], time.time()] for x in execute_programs]

            for i in itertools.count(0, 1):
                if not debug:
                    cur_time = time.time()

                    for x in execute_programs:
                        prog_time, prog, last_time = x
                        exec_prog = False
                        if prog_time > 0 and (cur_time - start_time) >= prog_time:
                            x[0] = 0
                            exec_prog = True
                        elif prog_time < 0 and (cur_time - last_time) >= -prog_time:
                            x[2] = cur_time
                            exec_prog = True

                        if exec_prog:
                            try:
                                exec(prog)
                            except Exception as e:
                                print("Unable to execute program: %s" % (prog))

                    if not is_reached_goal:
                        iter, iter_time, batch_size = model.train_one_iter()

                        loss_history = model.get_loss_history()
                        time_str = time.strftime("[%H:%M:%S]")
                        if iter_time >= 10:
                            loss_string = "{0}[#{1:06d}][{2:.5s}s][bs: {3}]".format(time_str, iter,
                                                                                    '{:0.4f}'.format(iter_time),
                                                                                    batch_size)
                        else:
                            loss_string = "{0}[#{1:06d}][{2:04d}ms][bs: {3}]".format(time_str, iter,
                                                                                     int(iter_time * 1000), batch_size)

                        if shared_state['after_save']:
                            shared_state['after_save'] = False
                            last_save_time = time.time()  # upd last_save_time only after save+one_iter, because plaidML rebuilds programs after save https://github.com/plaidml/plaidml/issues/274

                            mean_loss = np.mean([np.array(loss_history[i]) for i in range(save_iter, iter)], axis=0)

                            for loss_value in mean_loss:
                                loss_string += "[%.4f]" % (loss_value)

                            io.log_info(loss_string)

                            save_iter = iter
                        else:
                            for loss_value in loss_history[-1]:
                                loss_string += "[%.4f]" % (loss_value)

                            if io.is_colab():
                                io.log_info('\r' + loss_string, end='')
                            else:
                                io.log_info(loss_string, end='\r')

                        if model.get_target_iter() != 0 and model.is_reached_iter_goal():
                            io.log_info('Reached target iteration.')
                            model_save()
                            is_reached_goal = True
                            io.log_info('You can use preview now.')

                if not is_reached_goal and (time.time() - last_save_time) >= save_interval_min * 60:
                    model_save()
                    send_preview()

                if i == 0:
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
            print('Error: %s' % (str(e)))
            traceback.print_exc()
        break
    c2s.put({'op': 'close'})


class Preview:
    def __init__(self, c2s, s2c, preview_queue):
        self.c2s = c2s
        self.s2c = s2c
        self.preview_queue = preview_queue
        # self.wnd_name = "Training preview"
        # io.named_window(wnd_name)
        # io.capture_keys(wnd_name)

        self.previews = None
        self.loss_history = None
        self.selected_preview = 0
        self.update_preview = False
        self.is_showing = False
        self.is_waiting_preview = False
        self.show_last_history_iters_count = 0
        self.iter = 0
        self.batch_size = 1
        self.preview_min_height = 512
        self.preview_max_height = 1024
        self.close = False

    def get_preview(self):
        while not self.close:
            self.process_queue_items()
            self.update_preview_frame()

    def process_queue_items(self):
        if not self.c2s.empty():
            input = self.c2s.get()
            op = input['op']
            if op == 'show':
                self.is_waiting_preview = False
                self.loss_history = input['loss_history'] if 'loss_history' in input.keys() else None
                self.previews = input['previews'] if 'previews' in input.keys() else None
                self.iter = input['iter'] if 'iter' in input.keys() else 0

                if self.previews is not None:
                    self.resize_previews()
                    self.selected_preview = self.selected_preview % len(self.previews)
                    self.update_preview = True
            elif op == 'close':
                self.close = True
            elif op == 'update':
                self.update()
            elif op == 'next_preview':
                self.next_preview()
            elif op == 'change_history_range':
                self.change_history_range()

    def update_preview_frame(self):
        if self.update_preview:
            self.update_preview = False

            selected_preview_name = self.previews[self.selected_preview][0]
            selected_preview_rgb = self.previews[self.selected_preview][1]
            (h, w, c) = selected_preview_rgb.shape

            # HEAD
            head_lines = [
                '[s]:save [enter]:exit',
                '[p]:update [space]:next preview [l]:change history range',
                'Preview: "%s" [%d/%d]' % (selected_preview_name, self.selected_preview + 1, len(self.previews))
            ]
            head_line_height = 15
            head_height = len(head_lines) * head_line_height
            head = np.ones((head_height, w, c)) * 0.1

            for i in range(0, len(head_lines)):
                t = i * head_line_height
                b = (i + 1) * head_line_height
                head[t:b, 0:w] += imagelib.get_text_image((head_line_height, w, c), head_lines[i], color=[0.8] * c)

            final = head

            if self.loss_history is not None:
                if self.show_last_history_iters_count == 0:
                    loss_history_to_show = self.loss_history
                else:
                    loss_history_to_show = self.loss_history[-self.show_last_history_iters_count:]

                lh_img = models.ModelBase.get_loss_history_preview(loss_history_to_show, self.iter, self.batch_size, w,
                                                                   c)
                final = np.concatenate([final, lh_img], axis=0)

            final = np.concatenate([final, selected_preview_rgb], axis=0)
            final = np.clip(final, 0, 1)
            preview_pane = (final * 255).astype(np.uint8)
            retval, buffer = cv2.imencode('.jpg', preview_pane)
            # jpg_as_text = base64.b64encode(buffer)
            jpg_as_text = buffer.tostring()
            self.preview_queue.put(jpg_as_text)

    def resize_previews(self):
        preview_height = max((h for h, w, c in (im.shape for name, im in self.previews)))
        if preview_height > self.preview_max_height:
            preview_height = self.preview_max_height
        elif preview_height < self.preview_min_height:
            preview_height = self.preview_min_height

        # make all previews size equal
        for p in self.previews[:]:
            (preview_name, preview_rgb) = p
            (h, w, c) = preview_rgb.shape
            if h != preview_height:
                scale_factor = preview_height / float(h)
                self.previews.remove(p)
                self.previews.append((preview_name, cv2.resize(preview_rgb, (0, 0),
                                                          fx=scale_factor,
                                                          fy=scale_factor,
                                                          interpolation=cv2.INTER_AREA)))
        self.selected_preview = self.selected_preview % len(self.previews)

    def save(self):
        self.s2c.put({'op': 'save'})

    def exit(self):
        self.s2c.put({'op': 'close'})

    def update(self):
        if not self.is_waiting_preview:
            self.is_waiting_preview = True
            self.s2c.put({'op': 'preview'})

    def next_preview(self):
        self.selected_preview = (self.selected_preview + 1) % len(self.previews)
        self.update_preview = True

    def change_history_range(self):
        if self.show_last_history_iters_count == 0:
            self.show_last_history_iters_count = 5000
        elif self.show_last_history_iters_count == 5000:
            self.show_last_history_iters_count = 10000
        elif self.show_last_history_iters_count == 10000:
            self.show_last_history_iters_count = 50000
        elif self.show_last_history_iters_count == 50000:
            self.show_last_history_iters_count = 100000
        elif self.show_last_history_iters_count == 100000:
            self.show_last_history_iters_count = 0
        self.update_preview = True


def flask_thread(s2c, c2s, preview_queue):
    config = {
        "DEBUG": True,          # some Flask specific configs
        "CACHE_TYPE": "simple", # Flask-Caching related configs
        "CACHE_DEFAULT_TIMEOUT": 300
    }
    app = Flask(__name__)
    app.config.from_mapping(config)
    cache = Cache(app)
    template = """<html>
<head>
    <title>Video Streaming Demonstration</title>
</head>
<body>
<h1>Video Streaming Demonstration</h1>
<form action="/" method="post">
    <button name="save" value="save">Save</button>
    <button name="exit" value="exit">Exit</button>
    <button name="update" value="update">Update</button>
    <button name="next_preview" value="next_preview">Next preview</button>
    <button name="change_history_range" value="change_history_range">Change History Range</button>
</form>
<img src="{{ url_for('preview_image') }}">
</body>
</html>"""

    def gen():
        if not preview_queue.empty():
            frame = preview_queue.get()
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
            yield frame
            yield b'\r\n\r\n'

    @app.route('/', methods=['GET', 'POST'])
    def index():
        if request.method == 'POST':
            if 'save' in request.form:
                s2c.put({'op': 'save'})
            elif 'exit' in request.form:
                s2c.put({'op': 'close'})
            elif 'update' in request.form:
                c2s.put({'op': 'update'})
            elif 'next_preview' in request.form:
                c2s.put({'op': 'preview'})
            elif 'change_history_range' in request.form:
                c2s.put({'op': 'change_history_range'})
        return render_template_string(template)

    def queue_not_empty():
        return not preview_queue.empty()

    # @app.route('/preview_image')
    # @cache.cached(timeout=300, unless=queue_not_empty)
    # def preview_image():
    #     yield Response(preview_queue.get(),
    #                    mimetype='multipart/x-mixed-replace;boundary=frame')

    @app.route('/preview_image')
    @cache.cached(timeout=300, unless=queue_not_empty)
    def preview_image():
        return Response(preview_queue.get(), mimetype='image/jpeg')

    app.run(debug=True, use_reloader=False)


def main(args, device_args):
    io.log_info("Running trainer.\r\n")

    s2c = queue.Queue()
    c2s = queue.Queue()
    preview_queue = queue.Queue()

    e = threading.Event()
    thread = threading.Thread(target=trainerThread, args=(s2c, c2s, e, args, device_args))
    thread.start()

    e.wait()  # Wait for inital load to occur.

    flask_t = threading.Thread(target=flask_thread, args=(s2c, c2s, preview_queue))
    flask_t.start()

    preview = Preview(c2s, s2c, preview_queue)
    preview.get_preview()

