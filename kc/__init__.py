from logging.config import dictConfig
from flask import Flask, has_request_context, request
from flask_cors import CORS
from celery import Celery

from felfinder.models import db
from felfinder.config import SQLALCHEMY_DATABASE_URI

def make_celery(target_app):
    celery_instance = Celery(target_app.import_name,
                             backend=target_app.config['CELERY_RESULT_BACKEND'],
                             broker=target_app.config['CELERY_BROKER_URL'],
                             task_serializer = target_app.config['CELERY_TASK_SERIALIZER'],
                             event_serializer = target_app.config['CELERY_EVENT_SERIALIZER'],
                             accept_content = target_app.config['CELERY_ACCEPT_CONTENT'],
                             result_accept_content = target_app.config['CELERY_ACCEPT_CONTENT'])
    celery_instance.conf.update(target_app.config)
    taskbase = celery_instance.Task
    class ContextTask(taskbase):
        abstract = True
        def __call__(self, *args, **kwargs):
            with target_app.app_context():
                return taskbase.__call__(self, *args, **kwargs)
    celery_instance.Task = ContextTask
    return celery_instance

# Reference: http://flask.pocoo.org/snippets/35/
class ReverseProxied(object):
    '''Wrap the application in this middleware and configure the
    front-end server to add these headers, to let you quietly bind
    this to a URL other than / and to an HTTP scheme that is
    different than what is used locally.

    In nginx:
    location /myprefix {
        proxy_pass http://192.168.0.1:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Scheme $scheme;
        proxy_set_header X-Script-Name /myprefix;
        }

    :param app: the WSGI application
    '''
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        script_name = environ.get('HTTP_X_SCRIPT_NAME', '')
        if script_name:
            environ['SCRIPT_NAME'] = script_name
            path_info = environ['PATH_INFO']
            if path_info.startswith(script_name):
                environ['PATH_INFO'] = path_info[len(script_name):]

        scheme = environ.get('HTTP_X_SCHEME', '')
        if scheme:
            environ['wsgi.url_scheme'] = scheme
        return self.app(environ, start_response)



app = Flask(__name__)
CORS(app)
app.config.from_pyfile('config.py')
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI

db.init_app(app)
celery = make_celery(app)

import felfinder.routes
