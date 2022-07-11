import random
import pickle
import os
import subprocess as sp
import datetime
import string

from pprint import pformat
from itertools import tee

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
NUMBERS = "1234567890"
ALPHABET_plus_num= ALPHABET + NUMBERS


CORPUS_SIZE = 200
SET_NAME = str(CORPUS_SIZE) + "_large_text_files"


#mime_types = ['text/plain', 'image/jpeg', 'text/plain', 'application/vnd.google-apps.script+json','text/csv']

mime_types = ["application/vnd.google-apps.audio",
              "application/vnd.google-apps.document",
              "application/vnd.google-apps.drawing",
              "application/vnd.google-apps.file",
              "application/vnd.google-apps.form",
              "application/vnd.google-apps.fusiontable",
              "application/vnd.google-apps.map",
              "application/vnd.google-apps.photo",
              "application/vnd.google-apps.presentation",
              "application/vnd.google-apps.script",
              "application/vnd.google-apps.site",
              "application/vnd.google-apps.spreadsheet",
              "application/vnd.google-apps.unknown",
              "application/vnd.google-apps.video",
              "application/vnd.google-apps.drive-sdk"]

exts = ['txt', 'jpg']

args = ["id", "size", "mimeType", "md5Checksum", "quotaBytesUsed", "explicitlyTrashed",
        "kind", "ownedByMe", "isAppAuthorized", "trashed", "ext", "name", "iconLink",
        "version", "permissionIds", "viewersCanCopyContent", "writersCanShare",
        "webContentLink", "viewedByMeTime", "thumbnailLink",
        "copyRequiresWriterPermission", "lastModifyingUser", "shared",
        "webViewLink", "permissions", "modifiedByMe", "createdTime",
        "capabilities", "owners", "headRevisionId",
        "originalFilename", "hasThumbnail", "modifiedTime", "spaces",
        "starred", "viewedByMe", "thumbnailVersion", "parents"]


#exts = ['jpg', 'jpeg', 'png', 'tiff', 'tif', 'gif', 'bmp', '3gp', '3g2', 'avi', 'f4v', 'flv', 'm4v', 'asf', 'wmv', 'mpeg', 'mp4', 'qt', 'txt', 'doc', 'rtf', 'dotx', 'dot', 'odt', 'pages', 'tex', 'pdf', 'ps', 'eps', 'prn', 'odt', 'ott', 'odm', 'oth', 'ods', 'ots', 'odg', 'otg', 'odp', 'otp', 'odf', 'odb', 'odp', 'docx', 'html', 'xhtml', 'php', 'js', 'xml', 'war', 'ear' 'dhtml', 'mhtml', 'xls', 'xlsx', 'xltx', 'xlt', 'ods', 'xlsb', 'xlsm', 'xltm', 'ppt', 'pptx', 'pot', 'potx', 'ppsx', 'pps', 'pptm', 'potm', 'ppsm', 'key']

if __name__ == "__main__":
    with open("words.txt", "r") as f:
        wordlist = f.read().split('\n')
else:
    with open("utils_testing/words.txt", "r") as f:
        wordlist = f.read().split('\n')

wordlist = [item.replace('/', '') for item in wordlist]

def ascii_string(N):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

def rand_custom(length):
    return "".join([random.choice(ALPHABET_plus_num) for i in range(length)])

def rand_checksum():
    return "".join([random.choice(ALPHABET_plus_num) for i in range(32)])

def rand_id():
    return "".join([random.choice(ALPHABET_plus_num) for i in range(28)])

def rand_perm():
    return "".join([random.choice(NUMBERS) for i in range(20)])

def headRevId():
    return "".join([random.choice(ALPHABET_plus_num) for i in range(51)])

def file_stem(num_words = 5):
    return '_'.join(wordgen(random.randint(1, num_words)))

def file_name(ext):
    return file_stem() + "." + ext

def coin_flip():
    return random.choice([True, False])

def wordgen(length):
    return random.sample(wordlist, length)

def rand_color():
    return random.randint(0, 255)

def rand_color_scheme():
    return (rand_color(), rand_color(), rand_color())

def rand_web_labels():
    return [(rand_id(), None) for _ in range(random.randint(0, 10))]

def rand_color_item():
    return (*rand_color_scheme(), random.random(), random.random())

def rand_color_ret():
    return [rand_color_item() for _ in range(random.randint(0, 10))]

def rand_normal_labels():
    return [(rand_id(), random.random(), random.random())\
            for _ in range(random.randint(0, 10))]

def user_gen():

    user = {}

    user['me'] = coin_flip()
    user['kind'] = 'drive#user'
    user['displayName'] = ' '.join(wordgen(2))
    user['permissionId'] = rand_perm()
    user['emailAddress'] = user['displayName'].replace(' ', '.') + "@example.com"

    return user

def perm_user_gen():

    perm_user = {}

    perm_user['role'] = 'writer'
    perm_user['type'] = 'user'
    perm_user['deleted'] = coin_flip()
    perm_user['displayName'] = ' '.join(wordgen(2))
    perm_user['emailAddress'] = perm_user['displayName'].replace(' ', '.') + "@example.com"
    perm_user['kind'] = 'drive#permission'

    return perm_user

def capability():

    cap = {}

    cap['canTrash'] = coin_flip()
    cap['canChangeViewersCanCopyContent'] = coin_flip()
    cap['canRemoveChildren'] = coin_flip()
    cap['canChangeCopyRequiresWriterPermission'] = coin_flip()
    cap['canShare'] = coin_flip()
    cap['canListChildren'] = coin_flip()
    cap['canDelete'] = coin_flip()
    cap['canReadRevisions'] = coin_flip()
    cap['canAddChildren'] = coin_flip()
    cap['canMoveItemIntoTeamDrive'] = coin_flip()
    cap['canRename'] = coin_flip()
    cap['canEdit'] = coin_flip()
    cap['canCopy'] = coin_flip()
    cap['canComment'] = coin_flip()
    cap['canUntrash'] = coin_flip()
    cap['canDownload'] = coin_flip()

    return cap

def rand_date():
    yr = random.randint(2000, 2017)
    month = random.randint(1, 12)
    day = random.randint(1, 28)

    hr = random.randint(0, 23)
    minutes = random.randint(0, 59)
    seconds = random.randint(0, 59)
    ms = random.randint(0, 999)

    return "{0}-{1:02d}-{2:02d}T{3:02d}:{4:02d}:{5:02d}.{6:03d}Z".format(yr, month, day, hr,
                                                                         minutes, seconds, ms)

def rand_word_content(words_in_file):
    return ' '.join([random.choice(wordlist) for _ in range(words_in_file)])


def dummy_links(uid, files):
    return [((uid, rand_id()), \
             'https://drive.google.com/file/d/' + rand_id() + '/view?usp=drivesdk') \
            for _ in range(random.randint(0, 30))]

def gen_image_label():

    d = {}

    #d['error']

    image_l = []
    for i in range(random.randint(1, 6)):
        image_d = {}
        image_d['color'] = {}
        image_d['color']["red"], image_d['color']["green"], image_d['color']["blue"], \
            image_d['score'], image_d['pixelFraction'] = rand_color_item()
        image_l.append(image_d)

    image_stuff = {}
    image_stuff = {}
    image_stuff["dominantColors"] = {}
    image_stuff["dominantColors"]["colors"] = image_l

    d["imagePropertiesAnnotation"] = image_stuff

    web_ent = []

    for i in range(random.randint(1, 6)):
        web_e = {}
        web_e['description'] = ' '.join([random.choice(wordlist) for _ in range(10)])
        web_e['score'] = random.random()
        web_ent.append(web_e)


    best_guess = []
    for i in range(random.randint(1, 6)):
        best_g = {}
        best_g['label'] = random.choice(wordlist)
        best_g['languageCode'] = 'eng'
        best_guess.append(best_g)


    web_detect = {}
    web_detect["webEntities"] = web_ent
    web_detect["bestGuessLabels"] = best_guess

    d["webDetection"] = web_detect

    annots = []
    for i in range(random.randint(1, 6)):
        anno = {}
        anno['description'] = ' '.join([random.choice(wordlist) for _ in range(10)])
        anno['score'] = random.random()
        anno['topicality'] = random.random()
        annots.append(anno)

    d['labelAnnotations'] = annots

    d['fullTextAnnotation'] = {}
    d['fullTextAnnotation']['text'] = ' '.join([random.choice(wordlist) for _ in range(10)])

    return d

def dummy_image_responses(reqs):
    return {'responses': [gen_image_label() for _, _ in enumerate(reqs)]}

def canned_image_resp(reqs, fixture):
    return {'responses': [fixture for _, _ in enumerate(reqs)]}

class dummyVision:

    def __init__(self, fixture = None):
        self.fixture = fixture

    def batch_annotate_images(self, reqs):
        if self.fixture is not None:
            return canned_image_resp(reqs, self.fixture)
        else:
            return dummy_image_responses(reqs)

class dummyVisionWrapper:

    def __init__(self, fixture = None):
        self.fixture = fixture

    def ImageAnnotatorClient(self):
        if self.fixture is not None:
            return dummyVision(self.fixture)
        else:
            return dummyVision()


default_arg = {"id": rand_id(),
               "size": str(random.randint(0, 2e9)),
               "mimeType": random.choice(mime_types),
               "md5Checksum": rand_checksum(),
               "quotaBytesUsed": '0',
               "explicitlyTrashed": coin_flip(),
               "kind": 'drive#file',
               "ownedByMe": coin_flip(),
               "isAppAuthorized": coin_flip(),
               "trashed": coin_flip(),
               "ext": 'txt',
               "name": file_name('txt'),
               "iconLink": '',
               "version": '961',
               "permissionIds": [str(rand_perm()) for i in range(random.randint(0, 10))],
               "viewersCanCopyContent": coin_flip(),
               "writersCanShare": coin_flip(),
               "webContentLink": 'https://drive.google.com/uc?id=kappa&export=download',
               "viewedByMeTime": rand_date(),
               "thumbnailLink": 'https://lh3.googleusercontent.com/' + rand_id() + "-Q=s220",
               "copyRequiresWriterPermission": coin_flip(),
               "lastModifyingUser": user_gen(),
               "shared": coin_flip(),
               "webViewLink": 'https://drive.google.com/file/d/kappa/view?usp=drivesdk',
               "permissions": [perm_user_gen() for i in range(random.randint(0, 10))],
               "modifiedByMe": coin_flip(),
               "createdTime": rand_date(),
               "capabilities": capability(),
               "owners": user_gen(),
               "headRevisionId": headRevId(),
               "originalFilename": 'kappa',
               "hasThumbnail": coin_flip(),
               "modifiedTime": rand_date(),
               "spaces": ['drive'],
               "starred": coin_flip(),
               "viewedByMe": coin_flip(),
               "thumbnailVersion": '1',
               "parents": []}


class DummyDbxFile:

    def __init__(self):
        pass


class DummyDbxAllItems:

    def __init__(self, entries):
        self.entries = entries
        self.has_more = False

def file_gen_dropbox(base_name):

    f = DummyDbxFile()

    f.id = rand_id()

    ext = "txt"

    f.path_lower = base_name[1:] + "/" + file_name(ext)
    f.server_modified = datetime.datetime.strptime('2019-06-26T16:48:42Z', "%Y-%m-%dT%H:%M:%SZ")
    f.size = 50
    f.path_display = f.path_lower

    return f


def file_gen(base_name = "", to_write=True, parents=None, id=None, size=None, mimeType=None,
             md5Checksum=None,
             quotaBytesUsed=None, explicitlyTrashed=None, kind=None, ownedByMe=None,
             isAppAuthorized=None, trashed=None, ext=None, name=None, iconLink=None, version=None,
             permissionIds=None, viewersCanCopyContent=None, writersCanShare=None,
             webContentLink=None, viewedByMeTime=None, thumbnailLink=None,
             copyRequiresWriterPermission=None, lastModifyingUser=None, shared=None,
             webViewLink=None, permissions=None, modifiedByMe=None, createdTime=None,
             capabilities=None, owners=None, headRevisionId=None,
             originalFilename=None, hasThumbnail=None, modifiedTime=None, spaces=None,
             starred=None, viewedByMe=None, thumbnailVersion=None):

    outer = {}

    outer['mimeType'] = mimeType or random.choice(mime_types)
    outer['md5Checksum'] = md5Checksum or rand_checksum()
    outer['quotaBytesUsed'] = quotaBytesUsed or '0'
    outer['id'] = id or rand_id()
    outer['size'] = size or str(random.randint(0, 2e6))
    outer['explicitlyTrashed'] = False #explicitlyTrashed or coin_flip()
    outer['kind'] = 'drive#file'
    outer['ownedByMe'] = ownedByMe if ownedByMe is not None else coin_flip()
    outer['isAppAuthorized'] = isAppAuthorized if isAppAuthorized else coin_flip()
    outer['trashed'] = trashed if trashed is not None else coin_flip()

    ext = ext or random.choice(exts)
    # if coin_flip():
    #     del outer['size']

    outer['name'] = name or file_name(ext)
    outer['fileExtension'] = ext
    outer['iconLink'] = iconLink or ''
    outer['version'] = version or '961'
    outer['permissionIds'] = permissionIds if permissionIds is not None else \
                             [str(rand_perm()) for i in range(random.randint(0, 10))]
    outer['viewersCanCopyContent'] = viewersCanCopyContent if \
        viewersCanCopyContent is not None else coin_flip()
    outer['writersCanShare'] = writersCanShare if writersCanShare is not None else coin_flip()
    outer['webContentLink'] = webContentLink or \
                              'https://drive.google.com/uc?id=' + outer['id'] + '&export=download'
    outer['viewedByMeTime'] = viewedByMeTime or rand_date()
    outer['thumbnailLink'] = thumbnailLink or \
                             'https://lh3.googleusercontent.com/' + rand_id() + "-Q=s220"
    outer['copyRequiresWriterPermission'] = copyRequiresWriterPermission if \
        copyRequiresWriterPermission is not None else coin_flip()
    outer['lastModifyingUser'] = lastModifyingUser or user_gen()
    outer['shared'] = shared if shared is not None else coin_flip()
    outer['webViewLink'] = webViewLink or \
                           'https://drive.google.com/file/d/' + outer['id'] + '/view?usp=drivesdk'
    outer['permissions'] = permissions if permissions is not None else [perm_user_gen() for i in range(random.randint(0, 10))]
    outer['thumbnailVersion'] = thumbnailVersion or '1'
    outer['modifiedByMe'] = modifiedByMe if modifiedByMe is not None else coin_flip()
    outer['createdTime'] = createdTime or rand_date()
    outer['capabilities'] = capabilities or capability()
    outer['owners'] = owners or user_gen()
    outer['parents'] = parents or []
    outer['headRevisionId'] = headRevisionId or headRevId()
    outer['originalFilename'] = originalFilename or outer['name']
    outer['fullFileExtension'] = ext
    outer['hasThumbnail'] = hasThumbnail if hasThumbnail is not None else coin_flip()
    outer['modifiedTime'] = modifiedTime or rand_date()
    outer['spaces'] = ['drive']
    outer['starred'] = starred or coin_flip()
    outer['viewedByMe'] = viewedByMe if viewedByMe is not None else coin_flip()

    # a, mode = .7, 100
    # words_in_file = int(((np.random.pareto(a, 1) + 1) * mode)[0])
    # words_in_file = min(words_in_file, 100)

    # text = ' '.join([random.choice(wordlist) for _ in range(words_in_file)])

    # if to_write:
    #     with open(base_name + "/" + outer['name'], "w") as f:
    #         f.write(text)

    return outer


def real_file_gen(path_stem, fname, par_folder):

    """
    This takes a single file and returns an object
    corresponding to the format in which Google returns
    file information
    """

    # TODO: ext to mimetype


    f = file_gen(name = fname,
                 ext = get_ext(fname),
                 permissions = [],
                 permissionIds = [],
                 shared = False,
                 ownedByMe = True,
                 trashed = False,
                 mimeType = None,
                 parents = [par_folder['id']])

    if par_folder['id'] is None:
        del f['parents']



    f['full_path'] = path_stem + "/" + fname

    mod_time = os.path.getmtime(f['full_path'])
    format_str = '%Y-%m-%dT%H:%M:%S.%fZ'

    f['modifiedTime'] = datetime.datetime.utcfromtimestamp(int(mod_time)).strftime(format_str)

    return f

def real_fold_gen(pathname, dir_ids):

    """
    Generate the information about a real folder

    pathname (str): Full path to folder
    dir_ids (dict): matches full path to ID
    """

    folds = pathname.split('/')
    last_in_path = folds[-1]
    par_path = '/'.join(folds[:-1])
    par = dir_ids.get(par_path, None)

    f = file_gen(name = last_in_path,
                 mimeType = 'application/vnd.google-apps.folder',
                 ext = "",
                 permissions = [],
                 permissionIds = [],
                 shared = False,
                 ownedByMe = True,
                 trashed = False,
                 parents = [par] if par else [])

    if par is None:
        del f['parents']

    f['full_path'] = pathname
    dir_ids[pathname] = f['id']

    return f


def get_ext(fname):
    """ Returns extension on file, if any """
    if '.' not in fname:
        return ""
    return fname.split('.')[-1]

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def unique_rand(start, end):

    seen = set()

    while True:
        choice = random.randint(start, end)
        if choice not in seen:
            seen.add(choice)
            yield choice
        else:
            continue

def rand_wrapper(start, end):

    while True:
        yield random.randint(start, end)

def type_rand(get_unique):

    if get_unique:
        return unique_rand
    else:
        return rand_wrapper

def rand_split(l, num_partitions, unique):

    if len(l) <= 1:
        return [l]

    part_indices = [i for _, i in zip(range(num_partitions - 1), type_rand(unique)(1, len(l) - 1))]
    full_part_indices = [0] + part_indices + [len(l)]

    full_part_indices = sorted(full_part_indices)

    split_list = [l[i1:i2] for i1, i2 in pairwise(full_part_indices)]
    return iter(split_list)

def real_gen_file_tree(top_folder):

    """
    Pointed at top_folder, we want to walk through the subdirectory
    and return files and folders the way Google Drive would
    """

    fs = []
    dir_ids = {}

    for dirpath, dnames, fnames in os.walk(top_folder):

        fold_obj= {'id': None}
        if dirpath != top_folder:
            fold_obj = real_fold_gen(dirpath, dir_ids)
            fs.append(fold_obj)

        for f in fnames:
            fs.append(real_file_gen(dirpath, f, fold_obj))

    for f in fs:
        print(f['id'], f.get('parents', None), f['name'])

    return {'files': fs}

def gen_file_tree(num_files = 10, max_depth = 2, max_branches = 4,
                  base_name = "fake_file_tree", same_attribute = None, dbx = False):

    if (num_files < 1):
        return {'files': []}

    true_depth = random.randint(1, max_depth)

    files = gen_folder(base_name, 0, num_files, true_depth, max_branches, same_attribute, dbx)

    if not dbx:
        ret_d = {}
        ret_d['files'] = files

    else:
        ret_d = DummyDbxAllItems(files)

    return ret_d

def gen_folder(base_name, curr_level, num_files, max_depth, max_branches, same_attribute, dbx):

    UNIQUE = False

    name = base_name + "/" + rand_id() + "/"

    #sp.run(['mkdir', name])

    num_sub_branches = random.randint(0, max_branches)

    gen_file_args = {arg: None for arg in args}

    if same_attribute is not None:
        gen_file_args[same_attribute] = default_arg[same_attribute]

    folders = []

    if (num_sub_branches == 0) or (curr_level == max_depth):

        gen_file_args['parents'] = [name]
        if not dbx:
            files_in_folder = [file_gen(name, **gen_file_args) for _ in range(num_files)]
        else:
            files_in_folder = [file_gen_dropbox(name) for _ in range(num_files)]
        return files_in_folder

    else:

        num_in_here = random.randint(0, num_files)
        num_in_subfolders = num_files - num_in_here

        num_subfolders = random.randint(0, max_branches)

        for split in rand_split(range(num_in_subfolders), num_subfolders, UNIQUE):
            folders += gen_folder(name, curr_level + 1, len(split), \
                                  max_depth, max_branches, same_attribute, dbx)

        gen_file_args['parents'] = [base_name]
        if not dbx:
            files_in_folder = [file_gen(name, **gen_file_args) for _ in range(num_in_here)]
        else:
            files_in_folder = [file_gen_dropbox(name) for _ in range(num_in_here)]
        return folders + files_in_folder


if __name__ == "__main__":
    print(gen_file_tree(num_files = 100, max_depth = 6, max_branches = 6))
