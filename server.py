from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Save files to root directory
UPLOAD_FOLDER = os.getcwd()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'jpg', 'png'}

# Public URL base for backend
PUBLIC_URL_BASE = "http://localhost:8080"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part', 400
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Generate public URL
            url = f"{PUBLIC_URL_BASE}/{filename}"

            return redirect(url_for('upload_file'))
        return 'Invalid file type', 400

    files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith(('.mp4', '.jpg', '.png'))]

    def format_duration(seconds):
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes}:{str(seconds).zfill(2)}"

    file_items = []
    for f in files:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f)
        file_items.append(f'<li><a href="/{f}" target="_blank">{f}</a></li>')


    file_list_html = ''.join(file_items)

    return render_template_string(f'''
        <!doctype html>
        <title>Upload Image/Video</title>
        <h1>Upload a new file</h1>
        <form method="post" enctype="multipart/form-data">
          <input type="file" name="file" accept=".mp4,.jpg,.png">
          <input type="submit" value="Upload">
        </form>
        <h2>Uploaded Files</h2>
        <ul>{file_list_html}</ul>
    ''')

@app.route('/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
