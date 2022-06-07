import os
import re


def ls(path, exts=None, ignore_dot_underscore=True):
    """ Lists the directory and returns it sorted. Only the files with
    extensions in `exts` are kept. The output should match the output of Linux
    command "ls". It wraps os.listdir() which is not guaranteed to produce
    alphanumerically sorted items.
    Args:
        path (str): Absolute or relative path to list.
        exts (str or list of str or None): Extension(s). If None, files with
            any extension are listed. Each e within `exts` can (but does
            not have to) start with a '.' character. E.g. both
            '.tiff' and 'tiff' are allowed.
        ignore_dot_underscore (bool): Whether to ignore files starting with
            '._' (usually spurious files appearing after manipulating the
            linux file system using sshfs)
    Returns:
        list of str: Alphanumerically sorted list of files contained in
        directory `path` and having extension `ext`.
    """
    if isinstance(exts, str):
        exts = [exts]

    files = [f for f in sorted(os.listdir(path))]

    if exts is not None:
        # Include patterns.
        extsstr = ''
        for e in exts:
            extsstr += ('.', '')[e.startswith('.')] + '{}|'.format(e)
        patt_ext = r'.*({})$'.format(extsstr[:-1])
        re_ext = re.compile(patt_ext)

        # Exclude pattern.
        patt_excl = '^/'
        if ignore_dot_underscore:
            patt_excl = '^\._'
        re_excl = re.compile(patt_excl)

        files = [f for f in files if re_ext.match(f) and not re_excl.match(f)]

    return files
