source /path/to/virtualenv
screen -d -m gunicorn --worker-class gevent --log-level=debug -w 2 -b 0.0.0.0:5000 --access-logfile kc_log --error-logfile kc_log --timeout 150 --reload kc:app
screen -d -m /path/to/redis
sleep 5
# Hypothesis: We can afford to have more long_task workers than cores because many of the long tasks should be IO bound
#In this vein, we use -P eventlet on long_tasks to better support I/O bound tasks
screen -d -m celery  -c 4 -l INFO --logfile=celery.log -A felfinder.workers worker -Ofair --prefetch-multiplier 1 -Q long_tasks, -n long_tasks_worker
screen -d -m celery  -c 4 -l INFO --logfile=celery.log -A felfinder.workers worker --prefetch-multiplier 1 -Q quick_tasks, -n quick_tasks_worker
screen -d -m celery --autoscale=4,1 -l INFO --logfile=celery.log -A felfinder.workers worker -Q routing, -n routing_worker
screen -d -m celery -c 1 -l INFO --logfile=celery.log -A felfinder.workers worker -Q activity, -n activity_worker

screen -d -m /path/to/elasticsearch-7.6.1/bin/elasticsearch

TEST_SERVER=F
export TEST_SERVER
echo "All should be running"
