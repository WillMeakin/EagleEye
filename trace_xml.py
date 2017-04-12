#
# Project Eagle Eye
# Group 15 - UniSA 2015
# Gwilyn Saunders
# version 0.1.6
#
# A cheeky little script for tracing the robot movements from a topdown viewpoint.
#
# Example usage:
#	trace_xml.py point.xml -image_file <working_dir>\prefix
#
# Options:
#   -max_height      : to adjust the scaling (in pixels)
#   -width & -height : adjusts the bounds of the lab floor (in milimetres)
#   -codec           : specify a foucc video codec
#   -no_preview      : toggles previewing
#
#   -image_file : prefix dir\name for output images
#

import cv2, sys, os, time, numpy as np, random
from eagleeye import Memset, EasyArgs, Key
from eagleeye.display_text import *
import xml.etree.ElementTree as ET


def usage():
	print "trace_xml.py <xml file> {-image_file}"

def getCoords(fileName):
	coordsButton = {}
	coordsBack = {}
	for child in ET.parse(fileName).getroot():
		for frame in child.findall('frame'):
			if child.tag == 'buttonside':
				coordsButton[int(frame.get('num'))] = {'x': int(float(frame.find('vicon').get('x'))),
													   'y': int(float(frame.find('vicon').get('y'))),
													   'z': int(float(frame.find('vicon').get('z')))}
			elif child.tag == 'backside':
				coordsBack[int(frame.get('num'))] = {'x': int(float(frame.find('vicon').get('x'))),
													 'y': int(float(frame.find('vicon').get('y'))),
													 'z': int(float(frame.find('vicon').get('z')))}
			else:
				print("ERROR - in getFrameCoords_gt")
	return coordsButton, coordsBack

def main(sysargs):
	args = EasyArgs(sysargs)

	if 'help' in args:
		usage()
		return 0

	# arg sanity checks
	if len(args) < 2:
		usage()
		return 1

	if args.no_preview and not args.video_file and not args.image_file:
		print "if -no_preview is toggled, you must output an image"
		usage()
		return 1

	# default args
	height = args.height or 3000        # milimetres
	width = args.width or 9000          # milimetres
	max_height = args.max_height or 400  # pixels

	# working vars
	scale = float(max_height) / height
	img_h = int(height * scale)
	img_w = int(width * scale)

	# open window
	window_name = "Tracer"
	cv2.namedWindow(window_name)
	print img_w, img_h

	file = args[1]
	btnCoords, backCoords = getCoords(file)
	btnColour = (0, 0, 255)
	backColour = (255, 255, 0)

	def displayPoints(imgName, btnCoords, backCoords, axisA, axisB):
		baseFrame = np.zeros((img_h, img_w, 3), np.uint8)
		for frame in btnCoords:
			axisValA = int(float(btnCoords[frame][axisA]))
			axisValB = int(float(btnCoords[frame][axisB]))
			pt = (int(axisValA * scale), img_h - int(axisValB * scale))
			cv2.circle(baseFrame, pt, 1, btnColour, 1)
		for frame in backCoords:
			axisValA = int(float(backCoords[frame][axisA]))
			axisValB = int(float(backCoords[frame][axisB]))
			pt = (int(axisValA * scale), img_h - int(axisValB * scale))
			cv2.circle(baseFrame, pt, 1, backColour, 1)
		cv2.imshow(window_name, baseFrame)
		cv2.waitKey(0)

		if 'image_file' in args:
			cv2.imwrite(args.image_file+axisA.upper()+axisB.upper()+'.png', baseFrame)

	displayPoints(args.image_file, btnCoords, backCoords, 'x', 'y')
	displayPoints(args.image_file, btnCoords, backCoords, 'x', 'z')
	displayPoints(args.image_file, btnCoords, backCoords, 'y', 'z')

	cv2.destroyAllWindows()
	print "\ndone"
	return 0


if __name__ == "__main__":
	exit(main(sys.argv))
