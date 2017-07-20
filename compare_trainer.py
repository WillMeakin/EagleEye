#!/usr/bin/env python2
# 
# Project Eagle Eye
# Group 15 - UniSA 2015
# 
# Gwilyn Saunders, Kin Kuen Liu
# version 0.1.12
#
# Compares any corresponding image points to reprojected points
# And calculate and display the reprojection error
# 
# TODO: only compares one side of the trainer at a time because I'm too lazy to write dual support in
# 

import sys, cv2, numpy as np, time, os, csv
from eagleeye import BuffSplitCap, Xmlset, Xmltrainer, EasyArgs, EasyConfig, Key, marker_tool, Theta
from eagleeye.display_text import *
from math import sqrt

def usage():
    print "usage: compare_trainer.py <video file> <mapper xml> <trainer xml> {<mark_in> <mark_out> | -side <buttonside|backside|single> | -config <file> | -video_export <file> | -compare_export <file>}"

def main(sysargs):
    args = EasyArgs(sysargs)
    cfg = EasyConfig(args.cfg, group="compare_trainer")
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
    
    # determine sides
    side = Theta.resolve(args.side or 'buttonside')
    
    # open inouts files
    vid = BuffSplitCap(args[1], buff_max=cfg.buffer_size, side=Theta.Both)
    mapper_xml = Xmlset(args[2], offset=cfg.offset, offmode=Xmlset.offset_mode(cfg.offset_mode))
    trainer_xml = Xmltrainer(args[3], side=side)
    reprojerror_list = {}     # list of reprojection error of all frames
    lastframe = False
    
    # reject mapper_xml if it doesn't contain the trainer_target
    if cfg.trainer_target not in mapper_xml.data(0):
        print "Mapping file must contain training target:", cfg.trainer_target
        print "Target names in mapping file:", mapper_xml.data(0)
        print "Please specifiy correct target name in config"
        return 1
    
    # also reject if it's lens data doesn't match the trainer_xml
    _test_lens = mapper_xml.data(0)[cfg.trainer_target]['lens']
    if (_test_lens == Theta.NonDual and side != Theta.NonDual) \
            or (side != Theta.NonDual and _test_lens == Theta.NonDual):
        print "Mapping file doesn't match training file"
        print "map:", Theta.name(_test_lens), "trainer:", Theta.name(side)
        return 1
    
    # sync the video and xml
    vid.restrict(mark_in, mark_out)
    cropped_total = mark_out - mark_in
    mapper_xml.setRatio(cropped_total)
    
    # status
    print "ratio at:", mapper_xml.ratio()
    print "offset by:", cfg.offset, "in", cfg.offset_mode, "mode"
    print ""
    
    # open export (if specified)
    if args.video_export:
        in_fps  = vid.get(cv2.CAP_PROP_FPS)
        in_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        out_vid = cv2.VideoWriter(args.video_export,
                                  cv2.VideoWriter_fourcc(*cfg.fourcc),
                                  in_fps, vid.shape[:2])
        print "exporting to:", args.video_export
    else:
        cv2.namedWindow(window_name)
        print "interactive mode"
    
    
    # main loop
    while vid.isOpened():
        frame = vid.frame()
        reprojerror_list = compareReproj(frame, vid.at(), mapper_xml, trainer_xml, reprojerror_list, cfg)
        
        # export or navigate
        if args.video_export:
            sys.stdout.write("Reading Video Frame: {} / {}\r".format(vid.at(), mark_out))
            sys.stdout.flush()
            
            out_vid.write(frame)
            
            if vid.next():
                reprojerror_list = compareReproj(frame, vid.at(), mapper_xml, trainer_xml, reprojerror_list, cfg)
                mapper_xml.next()
            else:
                calReprojList(reprojerror_list)
                print "\nend of video"
                break
        else:
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(0)
            showAll = False     # toggle to show all training points in one frame
                
            # controls
            if key == Key.esc:
                print "\nexiting."
                break
            elif key == Key.right:
                if vid.next():
                    mapper_xml.next()
                else:
                    if lastframe is False:
                        print "\nEnd of video"
                        calReprojList(reprojerror_list)
                    lastframe = True
                    
            elif key == Key.left:
                if vid.back():
                    mapper_xml.back()
            
            elif key == Key.enter:
                while True:
                    reprojerror_list = compareReproj(frame, vid.at(), mapper_xml, trainer_xml, reprojerror_list, cfg)
                    cv2.imshow(window_name, frame)
                    
                    if vid.next():
                        mapper_xml.next()
                        frame = vid.frame()
                    
                    # break when over
                    else: break
                
                calReprojList(reprojerror_list)
                print "\nFast forwarded to end of video"
                
                if args.compare_export:
                    print "Output to file"
                    writeFile(args.compare_export, reprojerror_list)
                break
            
            elif key == Key.space:
                ''' TODO: display all training points, may be abandoned
                problems:
                frame is not refreshed after imshow
                if showAll is False:
                    displayTrainPts(frame, mark_in, mark_out, trainer_xml, cfg)
                    #
                    #except:
                    #    print "Error in displaying training points. Exiting..."
                    #    return 1
                    #
                    
                    showAll = True
                else:
                    frame, reprojerror_list = compareReproj(frame, vid.at(), mapper_xml, trainer_xml, reprojerror_list, cfg)
                    showAll = False
                '''
                pass
    
    # clean up
    vid.release()
    if args.video_export:
        out_vid.release()
    else:
        cv2.destroyAllWindows()
        
    return 0


# display original trainer point
def compareReproj(cvframe, vidframe_no, mapper_xml, trainer_xml, reprojerror_list, cfg):
    
    xmlframe_no = mapper_xml.at()
    sys.stdout.write("Reading Video Frame: {}".format(vidframe_no) + " | XML Frame number: {}\r".format(xmlframe_no))
    sys.stdout.flush()
    
    mapped_object = mapper_xml.data(xmlframe_no)[cfg.trainer_target]
    
    # Prepare Repojected Point
    pt1 = (int(float(mapped_object['box']['x'])), 
            int(float(mapped_object['box']['y'])))
    
    pt2 = (pt1[0] + int(float(mapped_object['box']['width'])), 
            pt1[1] + int(float(mapped_object['box']['height'])))
    
    centre = (int(float(mapped_object['centre']['x'])), 
                int(float(mapped_object['centre']['y'])))
    
    visible = int(mapped_object['visibility']['visible'])
    max_visible =  int(mapped_object['visibility']['visibleMax'])
    lens_side = mapped_object['lens']
    
    # Render Reprojected Point
    cv2.rectangle(cvframe, pt1, pt2, cfg.mapper_colour, 1)
    cv2.circle(cvframe, centre, 1, cfg.mapper_colour, 2)
    
    # Get trainer point if it exists at current frame
    vicon_txt = "VICON - x: ?, y: ?, z: ?"
    trainer_txt = "Trained Point: x: ?, y: ?"
    reprojErr_txt = "Reprojection Error: No Data"
    
    if vidframe_no in trainer_xml.data() and lens_side == trainer_xml.side:
        trainer_frame = trainer_xml.data()[vidframe_no]
        
        # fix buttonside trainer points
        rightside_offset = 960 if trainer_xml.side == Theta.Right else 0
        
        vicon_txt = "VICON - x: {:.4f}, y: {:.4f}, z: {:.4f}".format(
                            float(trainer_frame["vicon"]["x"]),
                            float(trainer_frame["vicon"]["y"]),
                            float(trainer_frame["vicon"]["z"]))
        trainer_txt = "Trained Point: x: {}, y: {}".format(
                            int(float(trainer_frame["plane"]["x"]) + rightside_offset),
                            int(float(trainer_frame["plane"]["y"])))
        
        # visualise trainer point
        trainer_pt = (int(float(trainer_frame["plane"]["x"]) + rightside_offset),
                        int(float(trainer_frame["plane"]["y"])))
        cv2.circle(cvframe, trainer_pt, 1, cfg.trainer_colour, 2)

        # Calculate Reprojection Error at this frame
        # Still display bad data reprojection error but not used in calculation of mean
        reprojErr = calReprojError(centre, trainer_pt)
        reprojErr_txt = "Reprojection Error: {} pixels".format(str(reprojErr))
        
        if cfg.ignore_baddata:
            if visible >= cfg.min_reflectors:
                reprojerror_list.update({vidframe_no: {
                                            "rms": reprojErr,
                                            "x1": trainer_pt[0],
                                            "y1": trainer_pt[1],
                                            "x2": pt1[0],
                                            "y2": pt1[1]}})
    
    # Display Object Info
    frameObj_Txt = "Frame: {} | Trained Object: {} | Comparing: {}".format(
                        vidframe_no, cfg.trainer_target, Theta.name(trainer_xml.side))
    reprojCentroid_txt = "Reprojected Centroid - x: {}, y: {}, on: {}".format(
                                                    centre[0], centre[1], Theta.name(lens_side))
    visibility_txt = "Visible: {} , Max Visible: {}".format(visible, max_visible)
    
    displayText(cvframe, frameObj_Txt, top=True)
    displayText(cvframe, reprojCentroid_txt)
    displayText(cvframe, visibility_txt)
    
    dataText = " - Good data!!"
    dataText_colour = (0, 255, 0) # green
    if visible < cfg.min_reflectors:           # for when there isn't a matching trainer
        dataText = " - Bad data!!"
        dataText_colour = (0, 0, 255) # red
        if cfg.ignore_baddata:
            dataText += " Ignored."
    
    displayText(cvframe, dataText, endl=True, colour=dataText_colour)
    displayText(cvframe, trainer_txt)
    displayText(cvframe, vicon_txt)
    displayText(cvframe, reprojErr_txt)
    
    return reprojerror_list


def writeFile(filepath, reprojList):
    with open(filepath, "wb") as csvFile:
        c = csv.writer(csvFile)
        for data in reprojList:
            c.writerow([data,
                        reprojList[data]["x1"],
                        reprojList[data]["x2"],
                        reprojList[data]["y1"],
                        reprojList[data]["y2"],
                        reprojList[data]["rms"]])


# calculate the difference between 2 points (manually picked & reprojected)
def calReprojError(img_pt, reproj_pt):
    if img_pt and reproj_pt is not None:
        if not(img_pt < 0) and not(reproj_pt < 0):
            return cv2.norm(img_pt, reproj_pt, cv2.NORM_L2) # Euclidean distance
    return -1.0


def calReprojList(reprojerror_list):
    # print reprojection error at end of vid
    print "\n"
    total_error = 0.0
    total_square_error = 0.0
    if len(reprojerror_list) > 0:
        for frm in reprojerror_list:
            total_error += float(reprojerror_list[frm]["rms"])
            
        arth_mean = total_error / len(reprojerror_list)
        #rms = sqrt(arth_mean)      # according to opencv calibrateCamera2
        for frm in reprojerror_list:
            total_square_error += float(reprojerror_list[frm]["rms"])**2
        rms = sqrt(total_square_error/len(reprojerror_list) - arth_mean**2)# sample standard deviation


        print "List of Reprojection Error:"
        for reproj in reprojerror_list:
            print "Frame: {} | Reprojection Error: {}".format(reproj, reprojerror_list[reproj]["rms"])
        print "\n"
        print "Number of matched training points: {}".format(len(reprojerror_list))
        print "RMS: {} pixels".format(rms)
        print "Arithmetical mean: {} pixels".format(arth_mean)
        print ""
    else:
        print "Reprojection Error: No Data"


# display all training points used
def displayTrainPts(frame, mark_in, mark_out, trainerxml, cfg):
    for i in range(mark_in +1, mark_out):
        if i in trainerxml.data():
            
            train_pt = trainerxml.data()[i]
            trainer_pt = (int(float(train_pt["plane"]["x"])),
                            int(float(train_pt["plane"]["y"])))
            # visualise trainer point
            cv2.circle(frame, trainer_pt, 1, cfg.trainer_colour, 2)
            

if __name__ == '__main__':
    exit(main(sys.argv))
