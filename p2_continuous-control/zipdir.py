import zipfile
import os

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

directory_to_zip = 'python'
zip_file_name = 'python_p2.zip'

zipf = zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED)
zipdir(directory_to_zip, zipf)
zipf.close()
