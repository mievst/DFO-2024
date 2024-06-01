from flask import Flask, request, send_from_directory, render_template, send_file
import os
from zipfile import ZipFile
import io
import json
from openpyxl import Workbook

from services import violation_service, export_service

app = Flask(__name__)
TIMECODES_FOLDER = "timecodes"
VIDEOS_FOLDER = 'videos'
DOWNLOAD_FOLDER = 'videos_for_download'
app.config['VIDEOS_FOLDER'] = VIDEOS_FOLDER
app.config['TIMECODES_FOLDER'] = TIMECODES_FOLDER

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
                filename = file.filename.replace(" ","_")
                path = os.path.join(app.config['VIDEOS_FOLDER'], filename)
                correct_path = path.replace(os.path.sep, '/')
                file.save(correct_path)

        return 'File uploaded successfully'
    else:
        return 'No file part'


@app.route('/upload/timecodes', methods=['POST'])
def upload_timecodes():
    data = request.get_json()
    if data is not None:
        path = os.path.join(app.config['TIMECODES_FOLDER'], "timecodes.json")
        correct_path = path.replace(os.path.sep, '/')
        with open(correct_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return 'OK', 200
    else:
        return 'Нет JSON данных в запросе', 400
    pass

@app.route('/show/violations', methods=['GET'])
def show_violations():
    path = os.path.join(app.config['TIMECODES_FOLDER'], "timecodes.json")
    correct_path = path.replace(os.path.sep, '/')
    if os.path.exists(correct_path):
        with open(correct_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        violations_dict = violation_service.get_violations_dict(data)
        violations_dict = violation_service.get_violation_dict_modified(violations_dict)
        return render_template('video-list.html',
                               name='VideoList',
                               violations=violations_dict), 200
    else:
        return 'На сервере нет данных по таймкодам', 400

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

@app.route('/videos/<filename>')
def videos(filename):
    return send_from_directory(app.config['VIDEOS_FOLDER'], filename)

@app.route('/export/xlsx', methods=['POST'])
def exportViolationsToXlsx():
    path = os.path.join(app.config['TIMECODES_FOLDER'], "timecodes.json")
    correct_path = path.replace(os.path.sep, '/')
    if os.path.exists(correct_path):
        if os.path.exists("temp_file.xlsx"):
            os.remove("temp_file.xlsx")
        wb,ws = export_service.export_violations_to_xlsx()
        temp_file = 'temp_file.xlsx'
        wb.save(temp_file)
        return send_file(temp_file, as_attachment=True, download_name='report.xlsx')
    else:
        return "Таймкодов в базе нет", 400

if __name__ == '__main__':
    app.run(debug=True)