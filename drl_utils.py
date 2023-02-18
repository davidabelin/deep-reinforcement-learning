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
from typing import Tuple
from math import floor
from torch.nn import Parameter


class VisualQNetwork(torch.nn.Module):
  def __init__(
    self,
    input_shape: Tuple[int, int, int],
    encoding_size: int,
    output_size: int
  ):
    """
    Creates a neural network that takes as input a batch of images (3
    dimensional tensors) and outputs a batch of outputs (1 dimensional
    tensors)
    """
    super(VisualQNetwork, self).__init__()
    height = input_shape[1]
    width = input_shape[2]
    initial_channels = input_shape[0]
    conv_1_hw = self.conv_output_shape((height, width), 8, 4)
    conv_2_hw = self.conv_output_shape(conv_1_hw, 4, 2)
    self.final_flat = conv_2_hw[0] * conv_2_hw[1] * 32
    self.conv1 = torch.nn.Conv2d(initial_channels, 16, [8, 8], [4, 4])
    self.conv2 = torch.nn.Conv2d(16, 32, [4, 4], [2, 2])
    self.dense1 = torch.nn.Linear(self.final_flat, encoding_size)
    self.dense2 = torch.nn.Linear(encoding_size, output_size)

  def forward(self, visual_obs: torch.tensor):
    conv_1 = torch.relu(self.conv1(visual_obs))
    conv_2 = torch.relu(self.conv2(conv_1))
    hidden = self.dense1(conv_2.reshape([-1, self.final_flat]))
    hidden = torch.relu(hidden)
    hidden = self.dense2(hidden)
    return hidden

  @staticmethod
  def conv_output_shape(
    h_w: Tuple[int, int],
    kernel_size: int = 1,
    stride: int = 1,
    pad: int = 0,
    dilation: int = 1,
  ):
    """
    Computes the height and width of the output of a convolution layer.
    """
    h = floor(
      ((h_w[0] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
    )
    w = floor(
      ((h_w[1] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
    )
    return h, w

###### 


###### 


###### 


###### 