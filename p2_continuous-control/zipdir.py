import zipfile
import os

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

### Zip
def zip(directory_to_zip = 'python', zip_file_name = 'python_p2.zip'):
	zipf = zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED)
	zipdir(directory_to_zip, zipf)
	zipf.close()


### Unzip
def unzip(file_path="python.zip", extract_path="/data"):
	# create a ZipFile object
	zip_file = zipfile.ZipFile(file_path, 'r')

	# extract all contents of the zip file to a specified directory
	zip_file.extractall(extract_path)

	# close the ZipFile object
	zip_file.close()
    
#zip(directory_to_zip = '/data/Reacher_One_Linux_NoVis', zip_file_name = 'reacher_one_class.zip')
#unzip(file_path="reacher_one_class.zip", extract_path="/data")