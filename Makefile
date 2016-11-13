
all: seq

seq:
	CUDA_VISIBLE_DEVICES='' KERAS_BACKEND=tensorflow ipython run.py
	#CUDA_VISIBLE_DEVICES='' KERAS_BACKEND=tensorflow python run.py

dist:
	mpirun -n 4 -x CUDA_VISIBLE_DEVICES='' -x KERAS_BACKEND=tensorflow python run.py
