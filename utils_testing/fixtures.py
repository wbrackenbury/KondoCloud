import os

from googleapiclient.discovery import build
from urllib.error import HTTPError

from utils_testing.test_google_auth_httplib2 import MockHttp, MockResponse
from utils_testing.test_google_auth_httplib2 import MockCredentials


def EMPTY_FUNC(*args, **kwargs):
    pass

def ERROR_FUNC(*args, **kwargs):
    raise ValueError

def HTTP_ERROR_FUNC(*args, **kwargs):
    with open(os.devnull, 'w') as of:
        raise HTTPError('http://localhost', 400, msg = 'suck it', hdrs = [], fp = of)

def HTTP_ERROR_FUNC_RETRY(*args, **kwargs):
    with open(os.devnull, 'w') as of:
        raise HTTPError('http://localhost', 429, msg = 'Try again', hdrs = [], fp = of)


def REPLACE_PARTIAL_F(f):

    return {'id': f.id,
            'hash': f.path_hash,
            'phash': f.parent_hash,
            'name': 'people',
            'mime': getattr(f, 'elf_mime_type', 'directory')}


def http_service_mock(uid, scope_level):
    creds = MockCredentials()
    resp = MockResponse(status = 401)

    mock_http = MockHttp(num_real_resps = 1, responses = [resp])

    return build('drive', 'v3', http=mock_http)
