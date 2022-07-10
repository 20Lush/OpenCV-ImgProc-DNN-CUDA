from __future__ import print_function
import os
import cv2


access_rights = 0o777 # or 0o755 idk

ex_file_path = '_stack/'

ex_video_path = '0.mp4'

skip_chunk = 31

os.mkdir('_stack/', access_rights)

## rips frames from provided video path, into provided file path.
def frame_gen(v_path, f_path): 

    ## print function parameters
    print('frame_gen f_path: ' + f_path)
    print('frame_gen V_path: ' + v_path)

    ## instantiate counters initialized to 0
    frame = 0 # cosmetic file counter that iterates by 1 for every write that isn't filtered out
    frame_ptr = 0 # the raw frame that is being looked at

    ## instantiate capture object 
    cap = cv2.VideoCapture(v_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) # debugging

    ## initializing frame grab. writes the first frame(numbered 0) 
    success, image = cap.read()
    cv2.imwrite(f_path + "frame%d.jpg" %(frame), image) #writes image to jpg numbered on the current frame counter

    ## loop as long as the cap.read() returns a valid image
    while success: 

        ## dynamic console line for debugging
        print(f"frames written: {frame} // frame_ptr: {frame_ptr} // total frames {total_frames}", sep='', end='\r', flush=True)
        
        ## frame skip clause. if the frame_ptr lands on an even element, it will refuse the write and skip over
        if frame_ptr % 2 == 0:
            frame_ptr += skip_chunk
            continue

        ## queue up a new image at element = frame_ptr
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ptr)
        success, image = cap.read()

        ## break out if the next frame grab fails 
        if success == False: 
            print('\nno next frame')
            break
        
        ## if the next frame grab is successful, iterate counters
        cv2.imwrite(f_path + "frame%d.jpg" %(frame), image)
        frame += 1
        frame_ptr += 1

frame_gen(ex_video_path, ex_file_path)
