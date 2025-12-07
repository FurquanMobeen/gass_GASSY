import os
from PIL import Image

# Set the directories to check
folders = [
    r'frames/13_44/nok',
    r'frames/13_44/ok',
    r'frames/14_32/nok',
    r'frames/14_32/ok',
    r'frames/14_43/nok',
    r'frames/14_43/ok',
    r'frames/14_55/nok',
    r'frames/14_55/ok'
]

# Minimum width and height
MIN_WIDTH = 500


def delete_small_images(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            try:
                delete_this = False
                width, height = None, None
                with Image.open(file_path) as img:
                    width, height = img.size
                    if width < MIN_WIDTH:
                        delete_this = True
                if delete_this:
                    print(f"Deleting {file_path} (size: {width}x{height})")
                    os.remove(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

def main():
    for folder in folders:
        if os.path.exists(folder):
            delete_small_images(folder)
        else:
            print(f"Folder does not exist: {folder}")

if __name__ == "__main__":
    main()
