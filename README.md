# PhotoAugmentation

This is an image augmentation program that will overlay a video ontop of an image and bring it to life!

This project would not have been possible without the tutoritals of [Murtaza's Workshop](https://www.youtube.com/c/MurtazasWorkshopRoboticsandAI/featured). Specifically, his tutorial series found [here](https://www.youtube.com/watch?v=7gSWd2hodFU) and [here](https://www.youtube.com/watch?v=nnH55-zD38I).

# Installation Requirements

If you would like to try this project out, I would first recommend creating a virtual environment using Python.

* `python3 -m venv virtual_env`

Then, activate the environment:

* `source virtual_env/bin/activate`

Once the virtual environment is activated, the follow packages are required:
* `pip3 install opencv-python`
* `pip3 install pillow`
* `pip3 install numpy`
* `pip3 install wand`

# Running the program

To run the program, all you have to do is run the `augment.py` file with:
* `python3 augment.py`

Once run, you'll be greeted with a windows asking you to upload and image and a video. The filetypes supported for images is `.jpg, .jpeg, .png and .heic`. The filetype supported for the video uploaded is `.mp4`.

Note: When uploading the image and the video, they must be name EXACTLY the same name for the program to work.

When the upload is complete, click the `Continue` button to use the program. Hold the image you uploaded up to the camera (I found the program does work best while using your mobile phone for your computers webcam) and the video chosen should be augmented ontop of the image!

