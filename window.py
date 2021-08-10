#!/usr/bin/env python3

import utils
import threading
import tkinter as tk
import tkinter.scrolledtext as tksc
import PIL.Image, PIL.ImageTk
import cv2

CAMERA = 0
DISPLAY_RATE = 80 # ms between images

class Window:

    def __init__(self, root, video_stream) -> None:

        self.root = root
        self.video_stream = video_stream
        self.width = self.video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.play_video = True

        # Create Image Canvas
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height)
        self.canvas.pack()

        # Start Thread to Update Images
        self.vid_thread = threading.Thread(target=self.update_stream)
        self.vid_thread.start()

    def update_stream(self) -> None:

        # Check if User has Paused Feed
        if self.play_video:
            ret, frame = utils.read_frame(self.video_stream) # Read Most Recent Frame
            # Get Text from Frame
            utils.bound_letters(frame, 
                utils.find_letters(frame), False
            ) 
            self.display.delete(1.0, tk.END) # Clear Output Console
            # Parse Letter Text
            self.display.insert(tk.END, '\n'.join(
                [''.join([letter.letter for letter in line]) for line in text][::-1]
            )) 
            self.display.update() # Update Output Console Text
            # Update Video if Frame is Opened
            if self.video_stream.isOpened() and ret:
                # Update Frame Image
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame)) 
                # Put Image on Canvas
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW) 
        # Update Again after DISPLAY_RATE
        self.root.after(DISPLAY_RATE, self.update_stream) 

def main() -> None:

    root = tk.Tk() # Get Root TkInter Object
    Window(root, cv2.VideoCapture(CAMERA)) # Make Window
    root.mainloop() # Run Main Loop

# RUN APPLICATION 
if __name__ == '__main__':
    main() # Run Main Function
    