rm kc_log
rm celery.log
source /path/to/virtualenv
python create_prod_tables.py
curl -XDELETE localhost:9200/*
