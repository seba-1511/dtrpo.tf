
all: seq

seq:
	CUDA_VISIBLE_DEVICES='' KERAS_BACKEND=tensorflow ipython run.py
	#CUDA_VISIBLE_DEVICES='' KERAS_BACKEND=tensorflow python run.py

dist:
	pass
