
# [7/6/2022] A script for gathering data from a 2560x1440p 60fps .mp4 file
#
# -> takes input from file
# -> saves every other frame as a picture (format tbd)
# -> keep a running index of how many files are deposited
# -> once deposition routine is either completed or stopped, leave behind some metadata
# ----> metadata: date, index.sizeof, source video file
# -> make these "stacks" easily indexible. have the unfinished stacks labeled "UNFINISHED"
# ----> script will look at most recent stack and check metadata to see if the current stack needs more data
# 
# -|| this is going to require cv probably

import os
from os.path import exists
import cv2 as cv

access_rights = 0o777 # or 0o755 idk

stack_capacity = 1000
n_ = 8 # n value for skipping. "skip until every nth value"



def frame_gen(v_path, f_path): # rips frames from provided video path, into provided file path. does not create file path. DOES ITS OWN METADATA CHECK

    cycles = 0
    if os.path.exists(f_path + "/" + "metadata.txt"):
        with open(f_path + "/" + "metadata.txt") as f:
            for line in f:
                cycles = int(line.strip('\n'))

    print('frame_gen f_path: ' + f_path)
    print('observed cycles: %d' %cycles)

    cap = cv.VideoCapture(v_path)
    success, image = cap.read()

    while success: # frame generator

        if cycles % n_ == 0: # ok idk how this works but it do be skipping most of the junk
            cycles += 1
            continue

        cv.imwrite(f_path + "/" + "frame%d.jpg" %(cycles*0.5), image)
        success, image = cap.read()

        if success == False:
            print('data exhausted')

        if cycles*0.5 == stack_capacity:
            success = False
            print("stack complete")
        else:
            cycles += 1

    if os.path.exists(f_path + "/" + "metadata.txt"):
        os.remove(f_path + "/" + "metadata.txt")

    with open(f_path + "/" + "metadata.txt", 'w') as f:
        f.write('%d' %(cycles*0.5))

def stack_gen():

    stacks = 0 #init global stack counter
    videos = 0 #init global video counter

    if os.path.exists('stack%d' %stacks):
        curr_stack_path = "stack%d" %stacks
    else:
        os.mkdir('stack%d' %stacks, access_rights)
        curr_stack_path = "stack%d" %stacks

    while os.path.exists('%d.mp4' %videos):
        
        print('stack_gen stacks: %d' %stacks)

        curr_video_path = '%d.mp4' %videos
        curr_stack_path = 'stack%d' %stacks

        if os.path.exists('stack%d' %stacks):
            curr_stack_path = "stack%d" %stacks
        else:
            os.mkdir('stack%d' %stacks, access_rights)
            curr_stack_path = "stack%d" %stacks

        frame_gen(curr_video_path, curr_stack_path) # deposit frames from selected video to current stack

        stacks += 1

        if os.path.exists(curr_stack_path + "/" + "metadata.txt"):
            with open(curr_stack_path + "/" + "metadata.txt") as f:
                for line in f:
                    cycles_check = int(line.strip('\n'))
                    if cycles_check != stack_capacity:
                        print('stack incomplete! appending incomplete stack with next video')
                        stacks -= 1 # dont iterate on stack if it isn't complete. frame_gen will automatically append on it and release the stack if the stack reaches max size
                        videos += 1 # however, move to next video
    
    print('no more videos')


stack_gen()
    

# print("base path is %s", base_path)

input("Press Enter to continue...")