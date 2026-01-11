import os
import uuid
from django.conf import settings
from django.core.exceptions import ValidationError


ALLOWED_EXTENSIONS = {".csv"}
MAX_FILE_SIZE_MB = 50


def save_uploaded_file_safely(uploaded_file) -> str:
    """
    Save uploaded file securely to MEDIA_ROOT/uploads/

    Returns:
        relative file path (str) to be stored in DB

    Raises:
        ValidationError if file is invalid
    """

    original_name = uploaded_file.name
    _, ext = os.path.splitext(original_name)

    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise ValidationError("Only CSV files are allowed.")

    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if uploaded_file.size > max_bytes:
        raise ValidationError(
            f"File too large. Max size is {MAX_FILE_SIZE_MB} MB."
        )

    upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    unique_name = f"{uuid.uuid4().hex}{ext}"
    full_path = os.path.join(upload_dir, unique_name)

    with open(full_path, "wb+") as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)

    return os.path.join("uploads", unique_name)





