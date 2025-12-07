import os
import random

nok_dir = r'frames/all-frames-data-set/nok'
nok_dir = os.path.normpath(nok_dir)
nok_files = [os.path.join(nok_dir, f) for f in os.listdir(nok_dir) if os.path.isfile(os.path.join(nok_dir, f))]

ok_dir2 = r'frames/all-frames-data-set/ok'
ok_dir2 = os.path.normpath(ok_dir2)
ok_files2 = [os.path.join(ok_dir2, f) for f in os.listdir(ok_dir2) if os.path.isfile(os.path.join(ok_dir2, f))]

nok_count = len(nok_files)
ok_count = len(ok_files2)


if nok_count > ok_count:
    to_remove = nok_count - ok_count
    files_to_delete = random.sample(nok_files, to_remove)
    for file_path in files_to_delete:
        print(f"Deleting from nok: {file_path}")
        os.remove(file_path)
else:
    to_remove = ok_count - nok_count
    files_to_delete = random.sample(ok_files2, to_remove)
    for file_path in files_to_delete:
        print(f"Deleting from ok: {file_path}")
        os.remove(file_path)
