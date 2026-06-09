from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Save files to root directory or a configured upload directory
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', os.getcwd())
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'jpg', 'jpeg', 'png'}

PUBLIC_URL_BASE = os.getenv('PUBLIC_URL_BASE')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_public_url(filename):
    if PUBLIC_URL_BASE:
        return f"{PUBLIC_URL_BASE.rstrip('/')}/{filename}"
    return url_for('uploaded_file', filename=filename, _external=True)


@app.route('/', methods=['GET'])
def index():
    files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f)]
    file_items = [f'<li><a href="/{f}" target="_blank">{f}</a></li>' for f in files]
    file_list_html = ''.join(file_items)

    return render_template_string(f'''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <title>Upload Image/Video</title>
        </head>
        <body>
            <h1>Upload a new file</h1>
            <form method="post" action="/upload" enctype="multipart/form-data">
              <input type="file" name="file" accept=".mp4,.jpg,.jpeg,.png" required>
              <input type="submit" value="Upload">
            </form>
            <h2>Uploaded Files</h2>
            <ul>{file_list_html}</ul>
        </body>
        </html>
    ''')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        url = get_public_url(filename)
        return jsonify({'filename': filename, 'url': url}), 201

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/files', methods=['GET'])
def list_files():
    files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f)]
    return jsonify({'files': files})


    @app.route('/<path:filename>', methods=['DELETE'])
    def delete_file(filename):
        safe_filename = secure_filename(filename)
        if safe_filename != filename:
            return jsonify({'error': 'Invalid filename'}), 400
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        try:
            os.remove(filepath)
            return jsonify({'filename': safe_filename, 'deleted': True})
        except OSError as e:
            return jsonify({'error': str(e)}), 500
@app.route('/<path:filename>', methods=['GET'])
def uploaded_file(filename):
    safe_filename = secure_filename(filename)
    if safe_filename != filename:
        return jsonify({'error': 'Invalid filename'}), 400
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)