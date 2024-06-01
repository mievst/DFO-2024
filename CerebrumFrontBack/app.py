from flask import Flask, request, send_from_directory, render_template, send_file
import os
from zipfile import ZipFile
import io
import json
from model import ModelManager

app = Flask(__name__)
VIDEOS_FOLDER = 'videos'
DOWNLOAD_FOLDER = 'videos_for_download'
app.config['VIDEOS_FOLDER'] = VIDEOS_FOLDER

@app.route('/')
def index():
    return render_template('index.html', name='UploadVideosPage')

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_files = request.files.getlist('file')

    if len(uploaded_files) > 0:
        print(len(uploaded_files))
        if not os.path.exists('videos'):
            os.makedirs('videos')

        for file in uploaded_files:
            if file.filename == '':
                return 'No selected file'
            if file:
                filename = file.filename
                path = os.path.join(app.config['VIDEOS_FOLDER'], filename)
                correct_path = path.replace(os.path.sep, '/')
                file.save(correct_path)
        manager = ModelManager()
        predict = manager.predict_in_folder(app.config['VIDEOS_FOLDER'])
        with open('timecodes.json', 'w') as json_file:
            json.dump(predict, json_file)
        print('File uploaded successfully')
        return 'File uploaded successfully'
    else:
        return 'No file part'


@app.route('/upload/timecodes', methods=['POST'])
def upload_timecodes():
    # Получаем JSON данные из тела запроса
    data = request.get_json()

    # Проверяем, что данные не пустые
    if data is not None:
        # Обрабатываем данные (например, выводим их в консоль)
        print(data)
        return 'JSON данные получены', 200
    else:
        return 'Нет JSON данных в запросе', 400

@app.route('/show/<filename>', methods=['GET'])
def show_file(filename):
    return send_from_directory(app.config['VIDEOS_FOLDER'], filename)

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    path = os.path.join(app.config['VIDEOS_FOLDER'], filename)
    correct_path = path.replace(os.path.sep, '/')
    return send_file(correct_path, as_attachment=True)

@app.route('/download/all', methods=['GET'])
def download_all():
    videos = [f for f in os.listdir(VIDEOS_FOLDER)]
    video_paths = []
    for video in videos:
        path = os.path.join(app.config['VIDEOS_FOLDER'], video)
        correct_path = path.replace(os.path.sep, '/')
        video_paths.append(correct_path)

    zip_stream = io.BytesIO()

    with ZipFile(zip_stream, 'w') as zip_file:
        for file_path in video_paths:
            zip_file.write(file_path, os.path.basename(file_path))
    zip_stream.seek(0)

    return send_file(zip_stream, as_attachment=True, download_name='videos.zip')

if __name__ == '__main__':
    app.run(debug=True)