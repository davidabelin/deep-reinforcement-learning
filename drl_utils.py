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


###### 


###### 


###### 


###### 


###### 


###### 