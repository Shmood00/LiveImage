from tkinter import *
from tkinter.filedialog import asksaveasfilename, askopenfilenames
import os
from wand.image import Image as WandImage
import shutil
import sys
from PIL import Image
import cv2


class Menu:
    def __init__(self, master, orb, images_path, videos_path, images, images_names, videos, desList, kpList):
        self.master = master
        self.orb = orb
        self.images_path = images_path
        self.videos_path = videos_path
        self.images = images
        self.images_names = images_names
        self.videos = videos
        self.desList = desList
        self.kpList = kpList

        # Update images / image names list prior to showing GUI (if images/videos already exist in directory)
        self.imgs()

        # Update video dict prior to showing GUI (if videos already exist in directory)
        self.vid()

        # Get descriptors of images prior to showing GUI (if images already exist in directory)
        self.get_descriptors()

        self.master.title("Choose Image / Video")
        self.master.geometry("900x500")

        self.label_file = Label(
            self.master, text="Please choose and image / video to augment ontop of the image.", width=100, height=4, fg="blue", bg="white")
        self.label_file.grid(column=1, row=1)

        self.label_inst = Label(self.master, text="Note: The image and video uploaded must be saved with the SAME name.",
                                width=100, height=4, fg="blue", bg="white")
        self.label_inst.grid(column=1, row=2)

        self.label_image_list = Label(
            self.master, text=f"Images in folder: {len(self.list_photos(images_path))}", width=100, height=2, fg="black", bg="white")
        self.label_image_list.grid(column=1, row=3)

        self.label_vid_list = Label(
            self.master, text=f"Videos in folder: {len(self.list_videos(videos_path))}", width=100, height=2, fg="black", bg="white")
        self.label_vid_list.grid(column=1, row=4)

        if len(self.list_photos(images_path)) == 0 or len(self.list_videos(videos_path)) == 0:

            self.label_warning = Label(self.master, text="Warning: No images or videos found. Please choose at least 1 image and 1 video.",
                                       width=100, height=4, fg="red", bg="white")
            self.label_warning.place(relx=0.0, rely=1.0, anchor="sw")

        self.button_exp = Button(self.master, text="Choose Image",
                                 command=self.browse_photo, width=10, height=1)
        self.button_exp.place(relx=0.5, rely=0.5, anchor="center")

        self.button_video = Button(self.master, text="Choose Video",
                                   command=self.browse_video, width=10, height=1)
        self.button_exit = Button(
            self.master, text="Continue", command=self.master.quit, width=10, height=1)
        self.button_quit = Button(
            self.master, text="Quit", command=self.quit, width=10, height=1)

        self.button_video.place(relx=0.5, rely=0.55, anchor="center")
        self.button_exit.place(relx=0.5, rely=0.6, anchor="center")
        self.button_quit.place(relx=0.5, rely=0.65, anchor="center")

    def list_photos(self, images_path):
        """ List images in the ImageQuery directory """
        images_list = os.listdir(images_path)

        if ".DS_Store" in images_list:
            images_list.remove(".DS_Store")

        return images_list

    def list_videos(self, videos_path):
        """ List videos in VideoQuery directory  """
        videos_list = os.listdir(videos_path)
        if ".DS_Store" in videos_list:
            videos_list.remove(".DS_Store")
        return videos_list

    def browse_photo(self):
        """
        Browse for an image and move it to the ImageQuery directory
        """
        filename = askopenfilenames(title="Pick a file...", filetypes=[
            ("jpg", "*.jpg"), ("png", "*.png"), ("jpeg", "*.jpeg"), ("HEIC", "*.heic")])

        for file in filename:

            if '.heic' not in file:

                img = Image.open(file)
                rgb_img = img.convert("RGB")

                new_name = asksaveasfilename(
                    title="Rename image something you'll remember")
                new_name = os.path.split(new_name)[1]

                rgb_img.save(os.path.abspath(
                    self.images_path)+f"/{new_name}.jpg")

                img.close()
            else:
                img = WandImage(filename=file)
                img.format = 'jpg'
                new_name = asksaveasfilename(
                    title="Rename image something you'll remember")

                new_name = os.path.split(new_name)[1]

                img.save(filename=os.path.abspath(
                    self.images_path)+f"/{new_name}.jpg")
                img.close()

            # shutil.move(rgb_img, images_path)

            self.label_file.config(
                text=f"Moved {len(filename)} image(s) to the {os.path.abspath(self.images_path)} directory.")
            images_list = self.list_photos(self.images_path)
            self.label_image_list.config(
                text=f"Images in folder: {len(images_list)}")

            # Update the images and images_names lists
            self.imgs()

            # Get the descriptors of the uploaded image wih orb detector
            self.get_descriptors()

    def browse_video(self):
        """
        Browse for a video and move it to the VideoQuery directory
        """
        filename = askopenfilenames(
            title="Pick a file...", filetypes=[("mp4", "*.mp4")])

        for file in filename:

            new_name = asksaveasfilename(
                title="Video title must be same name as image file saved")
            new_name = os.path.split(new_name)[1]
            try:
                shutil.copyfile(file, os.path.abspath(
                    self.videos_path)+"/video_"+new_name+".mp4")

                # Update the videos dict
                self.vid()

            except:
                pass

        self.label_file.config(
            text=f"Moved {len(filename)} video(s) to the {os.path.abspath(self.videos_path)} directory.")
        videos_list = self.list_videos(self.videos_path)
        self.label_vid_list.config(
            text=f"Videos in folder: {len(videos_list)}")

    def imgs(self):
        """
        Update the images and images_names lists
        """
        images_list = self.list_photos(self.images_path)
        for cl in images_list:
            imgCur = cv2.imread(f"{self.images_path}/{cl}")

            self.images.append(imgCur)

            self.images_names.append(os.path.splitext(cl)[0])

        return self.images, self.images_names

    def vid(self):
        """ Load the uploaded video and pass it into a dictionary.
            Key: image name
            Value: uploaded video VideoCapture object
        """
        images_list = self.list_photos(self.images_path)
        for i in range(len(images_list)):
            target_video = cv2.VideoCapture(
                f"{self.videos_path}/video_{os.path.splitext(images_list[i])[0]}.mp4")

            self.videos[os.path.splitext(images_list[i])[0]] = target_video

        return self.videos

    def get_descriptors(self):
        """ Get descriptors and key points of the images """
        for img in self.images:

            kp, des = self.orb.detectAndCompute(img, None)
            self.desList.append(des)
            self.kpList.append(kp)

        # return [descriptor_list, kps_list]

    def quit(self):
        sys.exit()
