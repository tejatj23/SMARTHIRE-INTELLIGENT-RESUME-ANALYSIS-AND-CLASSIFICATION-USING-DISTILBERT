import os
from flask import Blueprint, request, render_template

resume_bp = Blueprint("resume", __name__)

UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@resume_bp.route("/upload", methods=["GET", "POST"])
def upload_resume():

    if request.method == "POST":
        file = request.files["resume"]

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            return "Resume uploaded successfully!"

    return render_template("upload.html")