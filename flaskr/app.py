from pathlib import Path
from flask import Flask, send_file, Response, render_template, render_template_string, request, g
# from flask_socketio import SocketIO


def create_flask_app(s2c, c2s, s2flask, args):
    app = Flask(__name__, template_folder="templates")
    model_path = Path(args.get('model_path', ''))
    filename = 'preview.jpg'
    preview_file = str(model_path / filename)

    def gen():
        frame = open(preview_file, 'rb').read()
        while True:
            try:
                frame = open(preview_file, 'rb').read()
            except:
                pass
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
            yield frame
            yield b'\r\n\r\n'

    def send(queue, op):
        queue.put({'op': op})

    def send_and_wait(queue, op):
        while not s2flask.empty():
            s2flask.get()
        queue.put({'op': op})
        while s2flask.empty():
            pass
        s2flask.get()

    @app.route('/', methods=['GET', 'POST'])
    def index():
        if request.method == 'POST':
            if 'save' in request.form:
                send(s2c, 'save')
                return '', 204
            elif 'exit' in request.form:
                send(c2s, 'close')
                request.environ.get('werkzeug.server.shutdown')()
                return '', 204
            elif 'update' in request.form:
                send_and_wait(c2s, 'update')
            elif 'next_preview' in request.form:
                send_and_wait(c2s, 'next_preview')
            elif 'change_history_range' in request.form:
                send_and_wait(c2s, 'change_history_range')
            # return '', 204
        return render_template('index.html')

    # @app.route('/preview_image')
    # def preview_image():
    #     return Response(gen(), mimetype='multipart/x-mixed-replace;boundary=frame')

    @app.route('/preview_image')
    def preview_image():
        return send_file(preview_file, mimetype='image/jpeg', cache_timeout=-1)

    return app






