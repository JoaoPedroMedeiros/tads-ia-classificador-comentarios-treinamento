python3 manage.py migrate

uwsgi --socket 0.0.0.0:8000 --buffer-size 8192 --protocol uwsgi --module wsgi:application
