
all: bench

seq:
	CUDA_VISIBLE_DEVICES='' KERAS_BACKEND=tensorflow ipython run.py
	#CUDA_VISIBLE_DEVICES='' KERAS_BACKEND=tensorflow python run.py

dist:
	mpirun -n 4 -x CUDA_VISIBLE_DEVICES='' -x KERAS_BACKEND=tensorflow python run.py

bench:
	CUDA_VISIBLE_DEVICES='' KERAS_BACKEND=tensorflow python benchmark.py -e bench --momentum 0.0 --max_path_length 5000 --timesteps_per_batch 15000 --n_iter 10 --env 'InvertedPendulum-v1' --solved 950.0

bench-dist:
	pass

bench-momentum:
	pass
