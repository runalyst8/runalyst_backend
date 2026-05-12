#!/bin/bash

# This script is now the ENTRYPOINT of the container.
# It runs the migrations and then starts the Gunicorn server.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Running database migrations..."
alembic upgrade head

echo "Migrations complete. Starting Gunicorn server..."
# This is the final command. It will start the web server, and the script
# will continue to run as long as the server is running.
gunicorn -w 1 -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:${PORT}
