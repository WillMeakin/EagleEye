#!/usr/bin/env python2
# 
# Project Eagle Eye
# Group 15 - UniSA 2015
# 
# Gwilyn Saunders
# version 0.2.18
#
# Reads video and two datasets xml files
# then draws them over the video for comparison
# 

import sys, cv2, numpy as np, time, os
from eagleeye import BuffSplitCap, Xmlset, EasyArgs, EasyConfig, Key, marker_tool
from eagleeye.display_text import *

def usage():
    print "usage: compare.py <video file> <xml MAP dataset> <xml ANT dataset> {<mark_in> <mark_out> | -config <file> | -export <file>}"

def draw(frame, obj, colour):
    pt1 = (int(float(obj['box']['x'])), 
            int(float(obj['box']['y'])))
    pt2 = (pt1[0] + int(float(obj['box']['width'])), 
            pt1[1] + int(float(obj['box']['height'])))
    
    centre = (int(float(obj['centre']['x'])), 
            int(float(obj['centre']['y'])))
    
    cv2.rectangle(frame, pt1, pt2, colour, 1)
    cv2.circle(frame, centre, 1, colour, 2)

def main(sysargs):
    args = EasyArgs(sysargs)
    cfg = EasyConfig(args.cfg, group="compare")
    window_name = "EagleEye Comparator"
    
    if "help" in args:
        usage()
        return 0
    
    # grab marks from args
    if len(args) > 5:
        mark_in = args[4]
        mark_out = args[5]
        
        # test integer-ness
        try: int(mark_in) and int(mark_out)
        except: 
            usage()
            return 1
    
    # or grab them from a marker tool
    elif len(args) > 3:
        ret, mark_in, mark_out = marker_tool(args[1], 50, window_name)
        if not ret:
            print "Not processing - exiting."
            return 1
    else:
        usage()
        return 1
    
    # open video files
    vid = BuffSplitCap(args[1], buff_max=cfg.buffer_size)
    xml1 = Xmlset(args[2])
    xml2 = Xmlset(args[3])
    
    # trim the video
    vid.restrict(mark_in, mark_out)
    
    # trim the CSV
    cropped_total = mark_out - mark_in
    xml1.setRatio(cropped_total)
    xml2.setRatio(cropped_total) # should be 1.0 (total frames should be the same as video)
    print 'xml1 ratio:', xml1.ratio()
    print 'xml2 ratio:', xml2.ratio()
    
    # open export (if specified)
    if 'export' in args:
        in_fps  = vid.get(cv2.CAP_PROP_FPS)
        in_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        out_vid = cv2.VideoWriter(args.export,
                                  cv2.VideoWriter_fourcc(*cfg.fourcc),
                                  in_fps, in_size)
    else:
        cv2.namedWindow(window_name)
    
    while vid.isOpened():
        sys.stdout.write(vid.status() + "\r")
        sys.stdout.flush()
        
        frame = vid.frame()
        
        # draw objects from each dataset
        notvisible = 3
        for name in xml1.data():# MAP data set (i.e. ground truth data)
            visible = int(xml1.data()[name]["visibility"]['visible'])
            visibleMax = int(xml1.data()[name]["visibility"]['visibleMax'])
            if visible > (visibleMax - notvisible):
                draw(frame, xml1.data()[name], cfg.xml1_colour)
        for name in xml2.data():# ANT dataset (i.e. vision system result)
            draw(frame, xml2.data()[name], cfg.xml2_colour)
        
        # print status to screen
        displayText(frame, vid.status(), top=True)
        displayText(frame, "{}: {}".format(os.path.basename(args[2]), xml1.status()), colour=cfg.xml1_colour)
        displayText(frame, "{}: {}".format(os.path.basename(args[3]), xml2.status()), colour=cfg.xml2_colour)
        
        # export or navigate
        if 'export' in args:
            sys.stdout.write("{}/{}\r".format(vid.at(), cropped_total))
            sys.stdout.flush()
            
            out_vid.write(frame)
            
            if vid.next():
                xml1.next()
                xml2.next()
            else:
                print "\nend of video"
                break
        else:
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(0)
            
            # controls
            if key == Key.esc:
                print "\nexiting."
                break
            elif key == Key.right:
                if vid.next():
                    xml1.next()
                    xml2.next()
            elif key == Key.left:
                if vid.back():
                    xml1.back()
                    xml2.back()
    
    # clean up
    vid.release()
    if 'export' in args:
        out_vid.release()
    else:
        cv2.destroyAllWindows()
    
    return 0
    
if __name__ == '__main__':
    exit(main(sys.argv))
