"""Verify slurp_real.tar.gz md5 and inspect its member list WITHOUT extracting.

Checks: md5 vs Zenodo; total uncompressed size of regular files; member types; unsafe paths
(absolute, '..', symlink/hardlink/device members); top-level directories; FLAC counts;
cross-check against headset/far-field files referenced by the 8eb1654 annotations.
Usage (CPU node): python -I scripts/slurp/slurp_tar_inspect.py TARBALL ANNOT_DIR OUT_JSON
"""
import collections
import hashlib
import json
import posixpath
import sys
import tarfile

EXPECTED_MD5 = '9efc0f058ced47bf5131c7cb2cade513'


def main(tar_path, annot, out):
    h = hashlib.md5()
    with open(tar_path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    md5 = h.hexdigest()
    rep = dict(tarball=tar_path, md5=md5, md5_expected=EXPECTED_MD5, md5_ok=md5 == EXPECTED_MD5)
    print('MD5', md5, rep['md5_ok'], flush=True)
    if not rep['md5_ok']:
        json.dump(rep, open(out, 'w'), indent=2); sys.exit(2)
    types = collections.Counter(); unsafe = []; top = collections.Counter()
    size = 0; flac = {}; ext = collections.Counter()
    with tarfile.open(tar_path, 'r:gz') as t:          # streaming listing only, no extraction
        for m in t:
            kind = ('file' if m.isfile() else 'dir' if m.isdir() else 'symlink' if m.issym()
                    else 'hardlink' if m.islnk() else 'other')
            types[kind] += 1
            name = m.name
            norm = posixpath.normpath(name)
            if name.startswith('/') or norm.startswith('..') or '/../' in f'/{norm}/' or kind in ('symlink', 'hardlink', 'other'):
                unsafe.append(dict(name=name, kind=kind, linkname=m.linkname))
            top[norm.split('/')[0]] += 1
            if m.isfile():
                size += m.size
                ext[posixpath.splitext(name)[1].lower()] += 1
                if name.endswith('.flac'):
                    flac[posixpath.basename(name)] = name
    ann_head, ann_far = set(), set()
    for split in ('train', 'devel', 'test'):
        for line in open(f'{annot}/{split}.jsonl'):
            for x in json.loads(line)['recordings']:
                (ann_head if x['file'].endswith('-headset.flac') else ann_far).add(x['file'])
    dirs = collections.Counter(posixpath.dirname(p) for p in flac.values())
    rep.update(member_types=dict(types), unsafe_members=unsafe[:50], unsafe_count=len(unsafe),
               top_level=dict(top), uncompressed_regular_bytes=size, uncompressed_gib=size / 2**30,
               extensions=dict(ext), flac_files=len(flac), flac_dirs=dict(dirs),
               flac_headset=sum(k.endswith('-headset.flac') for k in flac),
               annotated_headset=len(ann_head), annotated_headset_missing_in_tar=len(ann_head - set(flac)),
               annotated_far=len(ann_far), annotated_far_missing_in_tar=len(ann_far - set(flac)),
               tar_flac_not_in_annotations=len(set(flac) - ann_head - ann_far),
               example_member=next(iter(flac.values()), None))
    json.dump(rep, open(out, 'w'), indent=2)
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:4])
