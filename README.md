# KondoCloud

This repository contains research artifacts for the publication
["KondoCloud: Improving Information Management in Cloud Storage via Recommendations Based on File Similarity"](https://wbrackenbury.github.io/assets/kondocloud_final.pdf),
as well as study infrastructure for
["Files of a Feather Flock Together? Measuring and Modeling How Users Perceive File Similarity in Cloud Storage"](https://wbrackenbury.github.io/assets/flock_final.pdf).
Some follow-up investigations are also located in this repository.

A large portion of the base code for the repository interface was
forked from the open-source web-based file browser,
[elFinder](https://github.com/Studio-42/elFinder).

## Setup

After filling in relevant values in `config.py` (particularly for
`POSTGRES_USER`, `POSTGRES_PASS`, `POSTGRES_DB`, `SECRET_KEY` `SALT`,
`GOOGLE_OAUTH_CLIENT_ID` and `GOOGLE_OAUTH_CLIENT_SECRET`), and
installing requirements with `pip install -r requirements.txt`,
one can ingest a local file tree with:

```
python test_local_files.py --del-db --create-db --load-to-db --file-tree [path to folder]
```

And examine the subsequent interface with:

```
python test_local_files.py --launch-window
```
In order to launch the full system, one can use the utility scripts
`prod_startup.py` and `rm_prod.py` to start up and wind down the relevant
scripts. One can flash the database with `prod_clear_all.py`. One will
first need to replace the paths to Redis and Elasticsearch executables
to run these. Redis is needed as a message broker for Celery, but
Elasticsearch is only needed to enable the search functionality in the
KondoCloud interface.


## Security

This code accesses sensitive scopes on Google Drive. A verification process
by Google is required before accessing such sensitive scopes of other users.
However, one may test this code on their own Google Drive
(e.g., [here](https://developers.google.com/drive/api/quickstart/python)).

* Note: * Given that this is a research artifact and not maintained, many
vulnerabilities have been discovered in required libraries since the study
launch. As this code accesses sensitive data, it is highly recommended
to update the requirements before testing the code for oneself.
