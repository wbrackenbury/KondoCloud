source venv_new/bin/activate
celery -f -A kc.workers purge
ps a | grep 'redis\|gunicorn\|flask\|celery\|elastic' | awk '{print $1}' | xargs kill -9
