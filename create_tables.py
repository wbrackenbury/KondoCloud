import os
from felfinder.models import *
from felfinder import app, db
from sqlalchemy.exc import OperationalError

with app.app_context():

    assert os.path.exists('workspace/')
    assert os.path.exists('vector_reps/')

    NUM_ATTEMPTS = 15

    for i in range(NUM_ATTEMPTS):
        try:
            db.create_all()
            db.session.commit()
            quit()
        except OperationalError:
            time.sleep(5)



    #If we've reached here, we haven't exited, and we're likely about to hit an error
    db.create_all()
    db.session.commit()
