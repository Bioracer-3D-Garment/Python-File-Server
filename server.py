from flask import Flask, request, send_from_directory, jsonify, url_for
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', os.getcwd())
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'mp4', 'jpg', 'jpeg', 'png'}

PUBLIC_URL_BASE = os.getenv('PUBLIC_URL_BASE')


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_public_url(filename: str) -> str:
    if PUBLIC_URL_BASE:
        return f"{PUBLIC_URL_BASE.rstrip('/')}/{filename}"
    return url_for('uploaded_file', filename=filename, _external=True)


@app.route('/', methods=['GET'])
def index():
    files = [
        f for f in os.listdir(app.config['UPLOAD_FOLDER'])
        if allowed_file(f)
    ]

    file_items = ''.join(
        f'<li><a href="/files/{f}" target="_blank">{f}</a></li>'
        for f in files
    )

    return f"""
    <html>
    <body>
        <h1>Upload</h1>
        <form method="post" action="/upload" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <input type="submit" value="Upload">
        </form>

        <h2>Files</h2>
        <ul>{file_items}</ul>
    </body>
    </html>
    """


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({
            'error': 'missing file field',
            'received_keys': list(request.files.keys())
        }), 400

    file = request.files['file']

    if file is None:
        return jsonify({'error': 'file is null'}), 400

    if file.filename is None or file.filename.strip() == '':
        return jsonify({'error': 'empty filename'}), 400

    filename = secure_filename(file.filename)

    if not allowed_file(filename):
        return jsonify({
            'error': 'invalid file type',
            'filename': filename
        }), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)
    except Exception as e:
        return jsonify({
            'error': 'save failed',
            'message': str(e)
        }), 500

    url = get_public_url(filename)

    return jsonify({
        'filename': filename,
        'url': url
    }), 201


@app.route('/files', methods=['GET'])
def list_files():
    files = [
        f for f in os.listdir(app.config['UPLOAD_FOLDER'])
        if allowed_file(f)
    ]
    return jsonify({'files': files})


@app.route('/files/<path:filename>', methods=['GET'])
def uploaded_file(filename):
    safe_filename = secure_filename(filename)

    if safe_filename != filename:
        return jsonify({'error': 'invalid filename'}), 400

    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/files/<path:filename>', methods=['DELETE'])
def delete_file(filename):
    safe_filename = secure_filename(filename)

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)

    if not os.path.exists(filepath):
        return jsonify({'error': 'not found'}), 404

    try:
        os.remove(filepath)
        return jsonify({'deleted': True, 'filename': safe_filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=True)