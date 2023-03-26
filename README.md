# Spatial Frequency Extraction using Gradient-liked Operator on Multispectral (SFEGO_Color)
## PyCUDA and PyOpenCL Version
### Introduction
- SFEGO_Color is doing the SFEGO on each RGB channel of image and concat as a RGB image that to visualize spatial info in multispectral.

- This work is based on SFEGO single channel version:
    - https://github.com/CardLin/SFEGO_PyCUDA 
    - https://github.com/CardLin/SFEGO_PyOpenCL

- The spatial frequency in the RGB image or multispectral image contain spectrum info, and we can use spatial decomposition to see this info.

- The image sensor is not only cpature light reflect by object inside FOV but also the lighting condiction of the light source.

- When light source passthrough the lens of camera and the refraction is different for each wavelength. So the image may contain spectrum info in it.

- The SFEGO_Color can analysis the image to extract the spectrum info.

- The Sun Light contain full spectrum so the decompostion result contain rainbow color in our decomposition result.
![SunLight](test1.jpg_SFEGO_Color_R56.0(2.0x28).jpg)

- The LED Light contain Blue and Yellow spectrum due to the LED cell emit the Blue Light and hit the Yellow Phosphor to generate the White Light.
![LedLight](test2.jpg_SFEGO_Color_R26.0(2.0x13).jpg)


### Hardware Requirement
- Require NVIDIA GPU to execute CUDA Kernel Code

- Recommend to use NVIDIA GPU with 1GB+ VRAM (VRAM usage is depend on Image Size)


### Execution
- Choose which GPGPU architecture you want to use. (comment out the architecture you don't want to use)
    - import SFEGO_PyOpenCL as SFEGO_Backend
    - import SFEGO_PyCUDA as SFEGO_Backend

- By default, it will using PyOpenCL to run the SFEGO. (notice: Integrated GPU also can run PyOpenCL and may slower than Discrete GPU)

- python SFEGO_Color.py test1.jpg
- python SFEGO_Color.py test2.jpg