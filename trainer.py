#!/usr/bin/env python2
# 
# Project Eagle Eye
# Group 15 - UniSA 2015
# 
# Gwilyn Saunders & Kin Kuen Liu
# version 0.5.38
#
# Process 1:
#  Left/right arrow keys to navigate the video
#  [ and ] to mark in/out the flash
#  ENTER to confirm and start clicker (Process 2)
#  ESC to quit
#
# Process 2:
#  Left click to mark wand centre
#  Left/right arrow keys to navigate the video
#  BACKSPACE to remove a previous record
#  ENTER to save (write XML)
#  ESC to abort (don't write XML)
# 
# Note:
#   add 'mark in and out' values to args to skip 'Process 1'
#
#
# TODO:
#   - add zoomer to Process 2 (probs requires QT)
#   - resize video to screen size in Process 1
# 

import sys, cv2, numpy as np, time, os
from eagleeye import BuffSplitCap, Memset, Key, EasyConfig, EasyArgs, marker_tool, Theta
from eagleeye.display_text import *
from elementtree.SimpleXMLWriter import XMLWriter

def usage():
    print "usage: trainer.py <video file> <csv file> <data out file> {<mark_in> <mark_out> | --clicks <num_clicks> | --config <file>}"

# clicker status
class Status:
    stop   = 0
    record = 1
    skip   = 2
    back   = 3
    remove = 4
    wait   = 5
    still  = 6

def main(sysargs):
    # settings
    args = EasyArgs(sysargs)
    cfg = EasyConfig(args.config, group="trainer")
    max_clicks = args.clicks or cfg.default_clicks
    window_name = "EagleEye Trainer"
    
    if "help" in args:
        usage()
        return 0
    
    # grab video flash marks from args
    if len(args) > 5:
        mark_in = args[4]
        mark_out = args[5]
        
        # test integer-ness
        try: int(mark_in) and int(mark_out)
        except: 
            usage()
            return 1
        
    elif len(args) > 3:
        ret, mark_in, mark_out = marker_tool(args[1], cfg.buffer_size, window_name)
        if not ret:
            print "Not processing - exiting."
            return 1
    else:
        usage()
        return 1
    
    #How many frames between flashes?
    cropped_total = mark_out - mark_in
    print "video cropped at:", mark_in, "to", mark_out, "- ({} frames)".format(cropped_total)
    
    # clicking function
    def on_mouse(event, x, y, flags, params):
        # left click to mark
        if event == cv2.EVENT_LBUTTONDOWN:
            params['pos'] = (x, y)
            params['status'] = Status.record
            
        # right click to skip
        elif event == cv2.EVENT_RBUTTONDOWN:
            params['status'] = Status.skip
    
    # working variables
    params = {'status': Status.wait, 'pos': None}
    write_xml = False
    textstatus = ""
    dataQuality = 0   # 0 = good, >0 = bad/potentially bad
    
    # default right side (buttonside)
    if cfg.dual_mode:
        lens = Theta.Right
        trainer_points = {Theta.Right:{}, Theta.Left:{}}
    else: # both sides otherwise
        lens = Theta.Both
        trainer_points = {Theta.Both:{}}
    
    print "Minimum reflectors: {} | Ignore Negative xyz: {}".format(cfg.min_reflectors, cfg.check_negatives)
        
    # load video (again)
    in_vid = BuffSplitCap(args[1], crop=(0,0,0,0), rotate=BuffSplitCap.r0, buff_max=cfg.buffer_size)
    in_vid.restrict(mark_in, mark_out) #reels up to flash sync
    
    # load csv (with appropriate ratio)
    in_csv = Memset(args[2])
    in_csv.restrict() #reels up to flash sync
    #since Vicon records at higher "framerate" than camera, in_csv.next() jumps forwards by <vicon frame count>/<cam frame count> rows.
    #TODO: what about variable (vicon) framerate? No average of all within timstamp range?
    in_csv.setRatio(cropped_total)
    
    # test for marker data
    if len(in_csv.row()) < 10:
        print "This CSV contains no marker data!\nAborting."
        return 1
    
    # status
    print ""
    print "Writing to:", args[3]
    print "Number of clicks at:", max_clicks
    print ""
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse, params)
    
    # grab clicks (Process 2)
    while in_vid.isOpened():
        frame = in_vid.frame(side=lens)
        
        sys.stdout.write(in_vid.status() + " | Clicks {} / {}\r".format(
                                                len(trainer_points[lens]), max_clicks))
        sys.stdout.flush()
        
        # prepare CSV data, click data
        tx = float(in_csv.row()[2])
        ty = float(in_csv.row()[3])
        tz = float(in_csv.row()[4])
        rx = float(in_csv.row()[5])
        ry = float(in_csv.row()[6])
        rz = float(in_csv.row()[7])
        
        # data quality status
        visible = int(in_csv.row()[9])
        max_visible = int(in_csv.row()[8])
        
        # status text to write
        textrow = "VICON - x: {:.4f} y: {:.4f} z: {:.4f} | rx: {:.4f} ry: {:.4f} rz: {:.4f}".format(tx, ty, tz, rx, ry, rz)
        textquality = "Visible: {} , Max Visible: {}".format(visible, max_visible)
        textstatus = "{} | {}/{} clicks".format(in_vid.status(), len(trainer_points[lens]), max_clicks)
        
        if lens == Theta.Left:
            textstatus += " - back side"
        elif lens == Theta.Right:
            textstatus += " - button side"
        #else none, no lens split
        
        # if data is qualified bad, reduce timeout by one
        if dataQuality > 0:
            dataQuality -= 1
        
        if dataQuality == 0:
            dataStatus = " - Good data!!"
            dataStatus_colour = (0, 255, 0) # green
        else:
            dataStatus = " - Potentially bad data (wait {})".format(dataQuality)
            dataStatus_colour = (0, 255, 255) # yellow
        
        # Data tests
        # values must be above 0 and minimum reflectors
        if (cfg.check_negatives and (tx <= 0 or ty <= 0 or tz <= 0)) \
                or visible < cfg.min_reflectors:
            dataStatus = " - Bad data!!"
            dataStatus_colour = (0, 0, 255) # red
            
            if cfg.ignore_baddata:
                dataStatus += " Ignored."
                dataQuality = 1 + cfg.quality_delay #default quality_delay=2
        
        # draw the trainer dot (if applicable)
        if in_vid.at() in trainer_points[lens]:
            cv2.circle(frame, trainer_points[lens][in_vid.at()][0], 1, cfg.dot_colour, 2)
            cv2.circle(frame, trainer_points[lens][in_vid.at()][0], 15, cfg.dot_colour, 1)
        
        # draw text and show
        displayText(frame, textrow, top=True)
        displayText(frame, textquality)
        displayText(frame, textstatus)
        displayText(frame, dataStatus, endl=True, colour=dataStatus_colour)
        
        cv2.imshow(window_name, frame)
        
        # pause for input
        while params['status'] == Status.wait:
            key = cv2.waitKey(10)
            if key == Key.esc:      #don't write anything, exit
                params['status'] = Status.stop
            elif key == Key.enter:  #write groundtruth correspondences xml, exit
                write_xml = True
                params['status'] = Status.stop
            elif key == Key.right:  #next frame
                params['status'] = Status.skip
            elif key == Key.left:   #previous frame
                params['status'] = Status.back
            elif key == Key.backspace:  #remove click point in current frame
                params['status'] = Status.remove
            elif Key.char(key, '1') and cfg.dual_mode:
                params['status'] = Status.still #swap lens
                lens = Theta.Left
            elif Key.char(key, '2') and cfg.dual_mode:
                params['status'] = Status.still #swap lens
                lens = Theta.Right
            
        # catch exit status
        if params['status'] == Status.stop:
            print "\nprocess aborted!"
            break
        
        # write data
        if params['status'] == Status.record \
                and len(trainer_points[lens]) != max_clicks: # TODO: does this disable recording clicks on the last frame
                
            if dataQuality == 0:
                trainer_points[lens][in_vid.at()] = (params['pos'], in_csv.row()[2:5], in_csv.row()[8:10])
            params['status'] = Status.skip
        
        # or remove it
        elif params['status'] == Status.remove \
                and in_vid.at() in trainer_points[lens]:
            del trainer_points[lens][in_vid.at()]
            print "\nremoved dot"
        
        # load next csv frame
        if params['status'] == Status.skip:
            if in_vid.next():
                in_csv.next()
            else:
                write_xml = True
                print "\nend of video: {}/{}".format(in_vid.at() -1, mark_out -1)
                break
        
        # or load previous csv frame
        elif params['status'] == Status.back:
            if in_vid.back():
                in_csv.back()
        
        # reset status
        params['status'] = Status.wait
    
    # clean up
    cv2.destroyAllWindows()
    
    
    ## write xml
    if write_xml:
        out_xml = XMLWriter(args[3])
        out_xml.declaration()
        doc = out_xml.start("TrainingSet")
        
        # source information
        out_xml.start("video", mark_in=str(mark_in), mark_out=str(mark_out))
        out_xml.data(os.path.basename(args[1]))
        out_xml.end()
        out_xml.element("csv", os.path.basename(args[2]))
        
        # training point data
        for lens in trainer_points:
            if lens == Theta.Right:
                out_xml.start("buttonside", points=str(len(trainer_points[lens])))
            elif lens == Theta.Left:
                out_xml.start("backside", points=str(len(trainer_points[lens])))
            else: # non dualmode
                out_xml.start("frames", points=str(len(trainer_points[lens])))
            
            for i in trainer_points[lens]:
                pos, row, markers = trainer_points[lens][i]
                x, y = pos
                
                out_xml.start("frame", num=str(i))
                out_xml.element("plane", 
                                x=str(x), 
                                y=str(y))
                out_xml.element("vicon", 
                                x=str(row[0]), y=str(row[1]), z=str(row[2]))
                out_xml.element("visibility", 
                                visibleMax=str(markers[0]), 
                                visible=str(markers[1]))
                out_xml.end()
                
            out_xml.end() # frames
        
        # clean up
        out_xml.close(doc)
        
        print "Data was written."
    else:
        print "No data was written"
    
    print "\nDone."
    return 0

if __name__ == '__main__':
    exit(main(sys.argv))
