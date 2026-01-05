#!/usr/bin/env python
"""
Django's command-line utility for administrative tasks.

This file is the single entry point for:
- running the development server
- database migrations
- superuser creation
- management commands
- production WSGI/ASGI bootstrapping

Keep this file SIMPLE and STABLE.
NO application logic should ever go here.
"""

import os
import sys


def main() -> None:
    """
    Main entry point for Django management commands.
    """

    # Ensure the correct settings module is loaded
    os.environ.setdefault(
        "DJANGO_SETTINGS_MODULE",
        "ad_click_fraud_classification.settings"
    )

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Django is not installed or not available in the current "
            "virtual environment.\n\n"
            "Did you:\n"
            "  1. Activate the virtual environment?\n"
            "  2. Install requirements.txt?\n"
            "  3. Use the correct Python interpreter?\n"
        ) from exc

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()

