import inspect

from itertools import zip_longest

from .faker import gen_file_tree, ascii_string

# Itertools recipe: https://docs.python.org/3/library/itertools.html
def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

class ExecuteSignature:

    """

    Since we're mocking all this, each method normally returns an HTML object to execute.
    However, since we're not implementing this over HTTP, we want to replace that

    """

    def __init__(self, f):
        self.func = f
        self.result = None
        self.headers = {}

    def __call__(self, *args, **kwargs):
        if not self.result:
            self.result = self.func(*args, **kwargs)
        return self

    def execute(self):
        return self.result


def ex_signature(Cls):

    """
    A class wrapper that attaches an ExecuteSignature to each of a class's public methods.
    """

    class NewCls:

        def __init__(self, *args, **kwargs):
            self.org_inst = Cls(*args, **kwargs)

        def __getattribute__(self, s):

            try:
                att = super(NewCls,self).__getattribute__(s)
                return att
            except AttributeError:
                pass

            att = self.org_inst.__getattribute__(s)

            if inspect.ismethod(att) and '_' != s[0]: # Is method, and not a builtin
                return ExecuteSignature(att)
            else:
                return att


    return NewCls

@ex_signature
class FileService:

    """

    Implements the portion of the Resource object that is responsible for handling
    parts of file tasks

    TODO: remove the fixed page size from init and list

    """

    def __init__(self, full_files = [], byte_resp = bytearray(),
                 error_func = None, real_gen = False):
        self.full_files = full_files['files']
        self.file_pages = {}
        self.byte_resp = byte_resp
        self.error_func = error_func
        self.real_gen = real_gen

        groups = list(grouper(self.full_files, 100))
        groups[-1] = list(filter(lambda x: x is not None, groups[-1])) #Remove None-filled items

        for i, g in enumerate(groups):
            self.file_pages[str(i)] = g

    def list(self, fields='*', corpora='drive', corpus='', driveID='',
             includeItemsFromAllDrives=True, includeTeamDriveItems=True,
             orderBy='', pageSize=100, pageToken='0', q='', spaces='',
             supportsAllDrives=True, supportsTeamDrives=True, teamDriveID=''):

        """
        TODO: Figure out what the deal with FIELDS is
        """

        if self.error_func:
            self.error_func()

        nextPage = int(pageToken) + 1

        resp = {'files': self.file_pages[pageToken]}
        if nextPage < len(self.file_pages):
            resp['nextPageToken'] = str(nextPage)

        return resp

    def export_media(self, fileId, mimeType):

        if self.error_func:
            self.error_func()

        if self.real_gen:
            try:
                f_obj = next(filter(lambda x: x['id'] == fileId, self.full_files))
            except StopIteration: # Newly created file, no ID found
                print("No such ID found! {}".format(fileId))
                return self.byte_resp
            if 'full_path' not in f_obj:
                raise ValueError("If this is a real file tree, you should" + \
                                 "have the file path attached")
            with open(f_obj['full_path'], 'rb') as of:
                resp = of.read()
        else:
            resp = self.byte_resp

        return resp

    def get_media(self, fileId):

        if self.error_func:
            self.error_func()

        if self.real_gen:
            try:
                f_obj = next(filter(lambda x: x['id'] == fileId, self.full_files))
            except StopIteration: # Newly created file, no ID found
                print("No such ID found! {}".format(fileId))
                return self.byte_resp
            if 'full_path' not in f_obj:
                raise ValueError("If this is a real file tree, you should" + \
                                 "have the file path attached")
            with open(f_obj['full_path'], 'rb') as of:
                resp = of.read()
        else:
            resp = self.byte_resp

        return resp

    def update(self, fileId, addParents = [], removeParents = [], fields = ''):

        if self.error_func:
            self.error_func()

    def copy(self, fileId, body = {}, fields = ''):

        if self.error_func:
            self.error_func()

        return {'id': ascii_string(24)}

    def create(self, body = {}, fields = ''):

        if self.error_func:
            self.error_func()

        return {'id': ascii_string(24)}

    def get(self, fileId, fields = ''):
        if self.error_func:
            self.error_func()

        return {'id': fileId, 'parents': [ascii_string(24)]}


class FakeService:

    """

    Implements the Google Drive API at https://www.googleapis.com/discovery/v1/apis/drive/v3/rest

    Or, at least, part of it.

    """

    def __init__(self, num_files = 100, full_files = None,
                 byte_resp = bytearray(), error_func = None, real_gen = False):
        self.full_files = full_files or \
            gen_file_tree(num_files = num_files, max_depth = 6, max_branches = 6)
        self.byte_resp = byte_resp
        self.error_func = error_func
        self.real_gen = real_gen

    def files(self):
        return FileService(self.full_files, self.byte_resp,
                           self.error_func, self.real_gen)
