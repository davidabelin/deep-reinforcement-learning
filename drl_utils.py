###### widget bar to display progress
!pip install progressbar
import progressbar as pb

widget = ['training loop: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA() ]

timer = pb.ProgressBar(widgets=widget, maxval=episode).start()

###### Animated GIF
import imageio

def create_gif(images, gif_file_name, duration=0.5):
    imageio.mimsave(gif_file_name, images, duration=duration)

# Example usage
images = [image1, image2, image3, ...] # array of color image arrays (H, W, 3) with values 0-255
create_gif(images, 'animated.gif')


###### Zip a directory
import zipfile
import os

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

directory_to_zip = 'path/to/directory'
zip_file_name = 'directory.zip'

zipf = zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED)
zipdir(directory_to_zip, zipf)
zipf.close()


###### 


###### 


###### 


###### 


###### 