#!/usr/bin/env python2
#
# Project Eagle Eye
# Group 15 - UniSA 2015
# Gwilyn Saunders
# version 0.9.21
#
# Retrieves Vicon data via TCP sockets.
# Includes syncronized timestamp data via a R232 COM port.
#
# usage: python2 vicon_capture.py {--time <in minutes> | --config <file>}
#

from eagleeye import ViconSocket, Sleeper, EasyConfig, EasyArgs
from datetime import datetime
from serial import Serial
import csv, sys, os

# set arguments
args = EasyArgs()
time = args.time or 180
cfg = EasyConfig(args.config)
outpath = os.path.join(cfg.output_folder, datetime.now().strftime(cfg.date_format))

num_frames = int(time * cfg.framerate) + (cfg.flash_delay * 2)
flash_at = [cfg.flash_delay, num_frames - cfg.flash_delay]
sleeper = Sleeper(1.0 / cfg.framerate)

# data directory sanity check
if not os.path.exists(cfg.output_folder):
    os.makedirs(cfg.output_folder)

# start the serial listener
if cfg.run_serial:
    try:
        serial = Serial(cfg.serial_device, 9600)
        serial.setRTS(0) # set at zero
    except OSError:
        print "Couldn't open serial device", cfg.serial_device
        quit(1)
else:
    print "Not listening to serial"


# open Vicon client
client = ViconSocket(cfg.ip_address, port=cfg.port)
client.open()
subjects = client.get("getSubjects")[1:]
max_all = len(subjects) * 7 # seven items of data per object

# print status
print ""
print "Using config:", cfg._path
print "Running for", time, "seconds ({} frames)".format(num_frames)
print "Flash delay at:", cfg.flash_delay, " ({} seconds)".format(int(cfg.flash_delay / cfg.framerate))
print "Capturing at", cfg.framerate, "frames per second"
print "Recording these subjects:\n", ", ".join(subjects)
print ""

# open CSV files
csvfiles = []
csvwriters = {}
for sub in subjects:
    path = "{0}_{1}.csv".format(outpath, sub)
    f = open(path, 'wb')
    w = csv.writer(f, delimiter=cfg.output_delimiter, quoting=csv.QUOTE_MINIMAL)
    csvfiles.append(f)
    csvwriters[sub] = w

# main loop
for c in range(0, num_frames):
    sleeper.stamp()
    
    # run flash
    flash = "."
    if cfg.run_serial and c in flash_at:
        serial.setRTS(1)
        flash = "F"
        sys.stdout.write("\r - - - - - - - Flash!\r")
    else: 
        serial.setRTS(0)
    
    all = client.get("getAll")
    
    i = 1 # ignore index 0 (number of subjects)
    while i < max_all:
        csvwriters[all[i]].writerow([sleeper.getStamp(), flash] + all[i+1:i+7])
        i += 7
    
    # sleep until next timestamp
    sys.stdout.write("{}/{}\r".format(c, num_frames))
    sleeper.sleep("\r - - - - - - - - - Late!\r")
    sys.stdout.flush()
    
# clean up
for f in csvfiles: f.close()
client.close()
    
print "\nComplete."
exit(0)

# End of file
