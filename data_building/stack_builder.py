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

stack_capacity = 2000

last_frame = 0 # no touch

n_ = 100 # n value for skipping. "skip until every nth value"

skip_chunk = 1 # has to be odd?

def frame_gen(v_path, f_path): # rips frames from provided video path, into provided file path. does not create file path. DOES ITS OWN METADATA CHECK
    
    global last_frame
    cycles = last_frame
    frames = 0
    if os.path.exists(f_path + "/" + "metadata.txt"):
        with open(f_path + "/" + "metadata.txt") as f:
            for line in f:
                frames = int(line.strip('\n'))

    print('frame_gen f_path: ' + f_path)
    print('observed frames: %d' %frames)

    cap = cv.VideoCapture(v_path)
    #cap.set(cv.CAP_PROP_POS_FRAMES, cycles-1)
    success, image = cap.read()

    while success: # frame generator

        if cycles % n_ != 0: # ok idk how this works but it do be skipping most of the junk
            #print('skipped')
            cycles += skip_chunk
        elif cycles % n_ == 0:
            cv.imwrite(f_path + "/" + "frame%d.jpg" %(frames), image)
            success, image = cap.read()

            if success == False:
                print('data exhausted')

            if frames == stack_capacity:
                success = False
                last_frame = cycles
                print("stack complete")
                print('last_frame: %d' %last_frame)
            
            else:
                frames += 1
                cycles += 1

    if os.path.exists(f_path + "/" + "metadata.txt"):
        os.remove(f_path + "/" + "metadata.txt")

    with open(f_path + "/" + "metadata.txt", 'w') as f:
        f.write('%d' %cycles)

    print('frames: %d' %frames)

def stack_gen():

    stacks = 0 #init global stack counter
    videos = 0 #init global video counter

    if os.path.exists('_stack%d' %stacks):
        curr_stack_path = "_stack%d" %stacks
    else:
        os.mkdir('_stack%d' %stacks, access_rights)
        curr_stack_path = "_stack%d" %stacks

    while os.path.exists('%d.mp4' %videos):
        
        print('stack_gen stacks: %d' %(stacks+1))

        curr_video_path = '%d.mp4' %videos
        curr_stack_path = '_stack%d' %stacks

        if os.path.exists('_stack%d' %stacks):
            curr_stack_path = "_stack%d" %stacks
        else:
            os.mkdir('_stack%d' %stacks, access_rights)
            curr_stack_path = "_stack%d" %stacks

        frame_gen(curr_video_path, curr_stack_path) # deposit frames from selected video to current stack

        stacks += 1

        if os.path.exists(curr_stack_path + "/" + "metadata.txt"):
            with open(curr_stack_path + "/" + "metadata.txt") as f:
                for line in f:
                    frames_check = int(line.strip('\n'))

        if frames_check != stack_capacity:
            print('stack incomplete! appending incomplete stack with next video')
            print('frame check: %d' %frames_check)
            stacks -= 1 # dont iterate on stack if it isn't complete. frame_gen will automatically append on it and release the stack if the stack reaches max size
            videos += 1
            continue
    
    print('no more videos')

stack_gen()
    
# print("base path is %s", base_path)

input("Press Enter to continue...")