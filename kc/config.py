from collections import defaultdict

POSTGRES_USER = "FILLER"
POSTGRES_PASS = "FILLER"
POSTGRES_DB = "FILLER"
SQLALCHEMY_DATABASE_URI = "postgresql://{}:{}@localhost:5432/{}".\
    format(POSTGRES_USER, POSTGRES_PASS, POSTGRES_DB)

SQLALCHEMY_DB_TEST_URI = "postgresql://{}:{}@localhost:5432/{}".\
    format(POSTGRES_USER, POSTGRES_PASS, "test_new_swamp")

SQLALCHEMY_TRACK_MODIFICATIONS = False

SECRET_KEY = "secret"
SALT = "salt"

REDIS = "redis://localhost:####"

CELERY_BROKER_URL = REDIS
CELERY_RESULT_BACKEND = REDIS

BASIC_SCOPES = ['https://www.googleapis.com/auth/userinfo.profile',
                'https://www.googleapis.com/auth/drive.readonly',
                'https://www.googleapis.com/auth/drive.activity.readonly']

MINIMAL_SCOPES = ['https://www.googleapis.com/auth/userinfo.profile',
                  'https://www.googleapis.com/auth/drive.activity.readonly']

REPLAY_SCOPES = ['https://www.googleapis.com/auth/drive',
                 'https://www.googleapis.com/auth/userinfo.profile',
                 'https://www.googleapis.com/auth/drive.activity.readonly']

ACTIVITY_SCOPES = ['https://www.googleapis.com/auth/drive.activity.readonly']

TEST_SERVER = True
SWAMPNET_RECS = True

GOOGLE_OAUTH_CLIENT_ID="CLIENT_ID"
GOOGLE_OAUTH_CLIENT_SECRET="SECRET"


CELERY_ROUTES = {'felfinder.workers.populate_google': {'queue': 'long_tasks'},
                 'felfinder.workers.activity_logs': {'queue': 'long_tasks'},
                 'felfinder.workers.google_content_process': {'queue': 'quick_tasks'},
                 'felfinder.workers.collect_results': {'queue': 'quick_tasks'},
                 'felfinder.workers.replay_all': {'queue': 'quick_tasks'},
                 'felfinder.workers.post_proc_wrapper': {'queue': 'quick_tasks'},
                 'felfinder.workers.index_and_finalize_text': {'queue': 'long_tasks'},
                 'felfinder.workers.google_simils': {'queue': 'long_tasks'},
                 'felfinder.workers.set_user_complete': {'queue': 'quick_tasks'},
                 'felfinder.workers.goog_celery_proc': {'queue': 'quick_tasks'},
                 'felfinder.workers.recommend_preds': {'queue': 'quick_tasks'},
                 'felfinder.workers.simil_wrapper': {'queue': 'quick_tasks'},
                 'felfinder.workers.precompute_recs': {'queue': 'long_tasks'},
                 'felfinder.workers.simil_proc': {'queue': 'long_tasks'}}

CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_EVENT_SERIALIZER = "json"

QUALTRICS_CONSENT_LINK = "link1"
QUALTRICS_DEMO_LINK = "link2"
QUALTRICS_FULL_ACTION = "link3"
QUALTRICS_PARED_ACTION = "link4"
QUALTRICS_SUS_LINK = "link5"
QUALTRICS_EXPLAN_LINK = "link6"

PROLIFIC_PART_ONE_END = "https://app.prolific.co/submissions/complete?cc=COMPLETECODE"
PROLIFIC_PART_TWO_END = "https://app.prolific.co/"

# We limit the max amount of text we process due to the difficulty
# of sending the full text over the wire
TOP_TEXT_SIZE = int(9 * 1e7)

REC_ACT_DECAY = 10

ROOT_ID = "root"
ROOT_NAME = ROOT_ID
ROOT_PHASH = "l0_"
ROOT_PATH = ""

MAX_SELECT = 1000
LOCAL_IMG_FEATS = True
DO_BERT_REPR = False

PER_TIME_YIELD_RECS = 0.1
BINOMIAL_P = 0.1
MAX_RAND_RECS = 10
MAX_ACT_SAMPS_NEEDED = 20
GROUP_SAMP_SIZE = 3

THRESH = .65
DIST_THRESH = 0.15
MAX_RATE_CHANGE = 0.025
MIN_THRESH = 0.4
MAX_THRESH = 0.9
DEC_EXPONENT = 0.025

SWAMPNET_REP_NAME = "swampnet_model_path"
W2V_MODEL_PATH = "/path/to/GoogleNews-vectors-negative300.bin"

PARED_PROTOCOL = False

CLF_MAX_VALS = {'tfidf_sim': 1.000,
                'word_vec_sim': 1.000,
                'color_sim': 183.000,
                'obj_sim': 0.667,
                'token_simil': 1.000,
                'bin_simil': 1.000,
                'perm_simil': 1.000,
                'tree_dist': 15.000,
                'size_dist': 21.920,
                'bigram_simil': 1.0000,
                'last_mod_simil': 21.167}

text = ['txt', 'doc', 'docx', 'rtf', 'dotx', 'dot', 'odt',
        'pages', 'tex', 'pdf', 'ps', 'eps', 'prn', 'pages']
image = ['jpg', 'jpeg', 'png', 'tiff', 'tif',
         'gif', 'bmp', 'heic', 'tga', 'dae']
spreadsheet =  ['tsv', 'csv', 'xls', 'xlsx', 'xltx', 'xlt', 'ods',
                'xlsb', 'xlsm', 'xltm']
media = ['mtp', 'itlp','nef','asf', 'cbr', 'flv', 'djvu', 'mod',
         'mpg', 'cur', 'psd', '3gp', 'pak',
         'flac', 'heic', 'thm', 'grf', 'mkv',
         'fh10', 'wma', 'pdn', 'map', 'hip', 'ts', 'ogg',
         'itl', 'aep', 'avi', 'mov', 'mid', 'm4a', 'cdr',
         'ai', 'xpl', 'rgssad', 'srt', 'vdf', 'ico', 'fh11',
         'wav', 'aiff', 'feq', '3gpp', 'gc', 'ithmb', 'itdb',
         'm4v', 'm3u', 'wad', 'sfl', 'sfk', 'lvl', 'mp4', 'aif', 'mp3',
         'ipa', 'jps', 'indd', 'vfs', 'itc2', 'kmz', 'cr2', 'dss']
presentation = ['ppt', 'pptx', 'odp']

EXT_TO_TYPE = defaultdict(lambda: 'other')
for t in text:
    EXT_TO_TYPE[t] = 'text'
for t in image:
    EXT_TO_TYPE[t] = 'image'
for t in spreadsheet:
    EXT_TO_TYPE[t] = 'spreadsheet'
for t in media:
    EXT_TO_TYPE[t] = 'media'
for t in presentation:
    EXT_TO_TYPE[t] = 'presentation'
