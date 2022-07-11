from mixer.backend.flask import mixer
from faker import Faker

from felfinder.models import db, User, File, Folder

def blend_file():

    fake = Faker()

    name = fake.file_name()
    parent_id = fake.md5()
    parent_hash = fake.file_path()

    f = mixer.blend(File,
                    name = name,
                    path = fake.file_path(),
                    path_hash = fake.file_path(),
                    parent_id = parent_id,
                    parent_hash = parent_hash,
                    original_name = name,
                    original_parent_id = parent_id,
                    original_parent_hash = parent_hash,
                    file_type = 'document',
                    file_extension = fake.file_extension(),
                    last_modified = fake.unix_time(),
                    goog_mime_type = 'application/plain-text',
                    elf_mime_type = 'application/plain-text',
                    access_type = 'owner',
                    media_info = '',
                    trashed = False)

    return f

def blend_folder():

    fake = Faker()

    f = mixer.blend(Folder,
                    name = fake.file_name(),
                    path = fake.file_path(),
                    path_hash = fake.file_path(),
                    parent_id = fake.md5(),
                    parent_hash = fake.file_path(),
                    last_modified = fake.unix_time(),
                    access_type = 'owner',
                    media_info = '')

    return f

def blend_user(uid=None, access_token=None, refresh_token=None, prolific_id=None):

    fake = Faker()

    f = mixer.blend(User,
                    id=uid or fake.md5(),
                    access_token=access_token or fake.md5(),
                    refresh_token=access_token or fake.md5(),
                    prolific_id=prolific_id or fake.md5())

    return f
