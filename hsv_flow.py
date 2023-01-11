#!/usr/bin/env python

'''
example to show optical flow estimation using DISOpticalFlow

USAGE: dis_opt_flow.py [<video_source>]

Keys:
 1  - toggle spatial propagation of flow vectors
 2  - toggle temporal propagation of flow vectors
 3  - toggle wisp filter
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

import video
import random
from common import draw_str


def build_filters():
	filters = []
	ksize = 31
	for theta in np.arange(0, np.pi, np.pi / 16):
		kern = cv.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv.CV_32F)
		kern /= 1.5*kern.sum()
		filters.append(kern)
	return filters

def apply_filters(img, filters):
	accum = np.zeros_like(img)
	for kern in filters:
		fimg = cv.filter2D(img, cv.CV_8UC3, kern)
		np.maximum(accum, fimg, accum)
	return accum

def detect(img, cascade):
	rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
									 flags=cv.CASCADE_SCALE_IMAGE)
	if len(rects) == 0:
		return []
	rects[:,2:] += rects[:,:2]
	return rects

def draw_rects(img, rects, color):
	for x1, y1, x2, y2 in rects:
		cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

def draw_flow(img, flow, filters, rects, step=16):
	warp = warp_flow(img, flow)
	hsv = draw_hsv(flow)
	
	h, w = img.shape[:2]
	y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
	fx, fy = flow[y,x].T
	
	y += random.randint(0,1)
	x += random.randint(0,1)
	fy += random.randint(0,1)
	fx += random.randint(0,1)
	
	lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
	lines = np.int32(lines)
	vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
	cv.polylines(vis, lines, 0, (100, 130, 5))
	for (x1, y1), (_x2, _y2) in lines:
		if not ((y1-5 < _y2 < y1+5) and (x1-5 < _x2 < x1+5)):
			cv.circle(hsv, (x1, y1), 1, (130, 5, 80), -1)
			
	gray = img
	
	thrs1 = 2500
	thrs2 = 2000
	edge = cv.Canny(gray, thrs1, thrs2, apertureSize=5)
	
	hsv = np.uint8(hsv/2.)
	hsv[edge != 0] = (130, 80, 5)
	
	if filters:
		hsv = apply_filters(hsv, filters)
		
	draw_rects(hsv, rects, (0, 255, 0))
		
	return hsv


def draw_hsv(flow):
	h, w = flow.shape[:2]
	fx, fy = flow[:,:,0], flow[:,:,1]
	ang = np.arctan2(fy, fx) + np.pi
	v = np.sqrt(fx*fx+fy*fy)
	hsv = np.zeros((h, w, 3), np.uint8)
	hsv[...,0] = ang*(180/np.pi/2)
	hsv[...,1] = 255
	hsv[...,2] = np.minimum(v*4, 255)
	bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
	return bgr


def warp_flow(img, flow):
	h, w = flow.shape[:2]
	flow = -flow
	flow[:,:,0] += np.arange(w)
	flow[:,:,1] += np.arange(h)[:,np.newaxis]
	res = cv.remap(img, flow, None, cv.INTER_LINEAR)
	return res


def main():
	import sys
	print(__doc__)
	try:
		fn = sys.argv[1]
	except IndexError:
		fn = 0

	filters = build_filters()

	cam = video.create_capture(fn)
	_ret, prev = cam.read()
	prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
	use_spatial_propagation = False
	use_temporal_propagation = False
	use_filter = False
	inst = cv.DISOpticalFlow.create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
	inst.setUseSpatialPropagation(use_spatial_propagation)
	
	cascade = cv.CascadeClassifier(cv.samples.findFile("haarcascades/haarcascade_frontalface_alt.xml"))

	flow = None
	while True:
		_ret, img = cam.read()
		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		
		
		if flow is not None and use_temporal_propagation:
			#warp previous flow to get an initial approximation for the current flow:
			flow = inst.calc(prevgray, gray, warp_flow(flow,flow))
		else:
			flow = inst.calc(prevgray, gray, None)
		prevgray = gray
		
		eas = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		eas = cv.equalizeHist(eas)

		if use_filter:
			cv.imshow('flow', draw_flow(gray, flow, filters, rects = detect(eas, cascade)))
		else:
			cv.imshow('flow', draw_flow(gray, flow, None, rects = detect(eas, cascade)))

		ch = 0xFF & cv.waitKey(5)
		if ch == 27:
			break
		if ch == ord('1'):
			use_spatial_propagation = not use_spatial_propagation
			inst.setUseSpatialPropagation(use_spatial_propagation)
			print('spatial propagation is', ['off', 'on'][use_spatial_propagation])
		if ch == ord('2'):
			use_temporal_propagation = not use_temporal_propagation
			print('temporal propagation is', ['off', 'on'][use_temporal_propagation])
		if ch == ord('3'):
			use_filter = not use_filter
			print('wisp filter is', ['off', 'on'][use_filter])

	print('Done')


if __name__ == '__main__':
	print(__doc__)
	main()
	cv.destroyAllWindows()
