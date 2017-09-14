.PHONY: all bench finger dist seq bench-dist

# Standard TRPO:
#BENCH_PRE = vanilla
#MOMENTUM = 0.0

# *** With Momentum Applied to the params BEFORE the linesearch ***

#BENCH_PRE = mom_0.3
#MOMENTUM = 0.3

#BENCH_PRE = mom_0.7
#MOMENTUM = 0.7

#BENCH_PRE = mom_-0.3
#MOMENTUM = -0.3

# *** With Momentum Applied to the params AFTER the linesearch ***

#BENCH_PRE = post_mom_0.3
#MOMENTUM = 0.3

#BENCH_PRE = post_mom_0.1
#MOMENTUM = 0.1

#BENCH_PRE = post_mom_0.7
#MOMENTUM = 0.7

#BENCH_PRE = post_mom_-0.3
#MOMENTUM = -0.3

PRE = vanilla
MOMENTUM = 0.3
NREPLICAS = 8
BENCH_PRE = "dist_$(NREPLICAS)_$(PRE)_mom_$(MOMENTUM)"


all: 
	CUDA_VISIBLE_DEVICES='' KERAS_BACKEND=tensorflow python benchmark.py -e dev --momentum 0.0 --max_path_length 5000 --timesteps_per_batch 15000 --n_iter 300 --env 'InvertedPendulum-v1' --solved 950.0

seq:
	CUDA_VISIBLE_DEVICES='' KERAS_BACKEND=tensorflow ipython run.py
	#CUDA_VISIBLE_DEVICES='' KERAS_BACKEND=tensorflow python run.py

dist:
	mpirun -n 4 -x CUDA_VISIBLE_DEVICES='' -x KERAS_BACKEND=tensorflow python run.py

finger:
	CUDA_VISIBLE_DEVICES='' KERAS_BACKEND=tensorflow python benchmark.py -e bench --momentum 0.0 --max_path_length 5000 --timesteps_per_batch 50000 --n_iter 300 --env 'Finger-v1' --solved 950.0


bench:
	mpirun -n $(NREPLICAS) -x CUDA_VISIBLE_DEVICES='' -x KERAS_BACKEND=tensorflow python benchmark.py -e $(BENCH_PRE) --momentum $(MOMENTUM) --max_path_length 5000 --timesteps_per_batch 5000 --n_iter 300 --env 'SmallInvertedPendulum-v1' --solved 950.0
	mpirun -n $(NREPLICAS) -x CUDA_VISIBLE_DEVICES='' -x KERAS_BACKEND=tensorflow python benchmark.py -e $(BENCH_PRE) --momentum $(MOMENTUM) --max_path_length 5000 --timesteps_per_batch 5000 --n_iter 300 --env 'BigInvertedPendulum-v1' --solved 950.0
	mpirun -n $(NREPLICAS) -x CUDA_VISIBLE_DEVICES='' -x KERAS_BACKEND=tensorflow python benchmark.py -e $(BENCH_PRE) --momentum $(MOMENTUM) --max_path_length 5000 --timesteps_per_batch 5000 --n_iter 300 --env 'InvertedPendulum-v1' --solved 950.0
	mpirun -n $(NREPLICAS) -x CUDA_VISIBLE_DEVICES='' -x KERAS_BACKEND=tensorflow python benchmark.py -e $(BENCH_PRE) --momentum $(MOMENTUM) --max_path_length 5000 --timesteps_per_batch 15000 --n_iter 300 --env 'InvertedDoublePendulum-v1' --solved 9100.0
	#mpirun -n $(NREPLICAS) -x CUDA_VISIBLE_DEVICES='' -x KERAS_BACKEND=tensorflow python benchmark.py -e $(BENCH_PRE) --momentum $(MOMENTUM) --max_path_length 5000 --timesteps_per_batch 15000 --n_iter 300 --env 'Reacher-v1' --solved -3.75
	#mpirun -n $(NREPLICAS) -x CUDA_VISIBLE_DEVICES='' -x KERAS_BACKEND=tensorflow python benchmark.py -e $(BENCH_PRE) --momentum $(MOMENTUM) --max_path_length 5000 --timesteps_per_batch 25000 --n_iter 300 --env 'HalfCheetah-v1' --solved 4800.0
	#mpirun -n $(NREPLICAS) -x CUDA_VISIBLE_DEVICES='' -x KERAS_BACKEND=tensorflow python benchmark.py -e $(BENCH_PRE) --momentum $(MOMENTUM) --max_path_length 5000 --timesteps_per_batch 25000 --n_iter 300 --env 'Hopper-v1' --solved 3800.0
	#mpirun -n $(NREPLICAS) -x CUDA_VISIBLE_DEVICES='' -x KERAS_BACKEND=tensorflow python benchmark.py -e $(BENCH_PRE) --momentum $(MOMENTUM) --max_path_length 5000 --timesteps_per_batch 25000 --n_iter 300 --env 'Swimmer-v1' --solved 360.0
	mpirun -n $(NREPLICAS) -x CUDA_VISIBLE_DEVICES='' -x KERAS_BACKEND=tensorflow python benchmark.py -e $(BENCH_PRE) --momentum $(MOMENTUM) --max_path_length 5000 --timesteps_per_batch 50000 --n_iter 300 --env 'Ant-v1' --solved 6000.0
	mpirun -n $(NREPLICAS) -x CUDA_VISIBLE_DEVICES='' -x KERAS_BACKEND=tensorflow python benchmark.py -e $(BENCH_PRE) --momentum $(MOMENTUM) --max_path_length 5000 --timesteps_per_batch 50000 --n_iter 300 --env 'AmputedAnt-v1' --solved 6000.0
	mpirun -n $(NREPLICAS) -x CUDA_VISIBLE_DEVICES='' -x KERAS_BACKEND=tensorflow python benchmark.py -e $(BENCH_PRE) --momentum $(MOMENTUM) --max_path_length 5000 --timesteps_per_batch 50000 --n_iter 300 --env 'BigAnt-v1' --solved 6000.0
	mpirun -n $(NREPLICAS) -x CUDA_VISIBLE_DEVICES='' -x KERAS_BACKEND=tensorflow python benchmark.py -e $(BENCH_PRE) --momentum $(MOMENTUM) --max_path_length 5000 --timesteps_per_batch 50000 --n_iter 300 --env 'ExtendedAnt-v1' --solved 6000.0


dev_plot:
	#python plot_exp.py vanilla SmallInvertedPendulum-v1 BigInvertedPendulum-v1 InvertedPendulum-v1 InvertedDoublePendulum-v1 Hopper-v1 Swimmer-v1 Ant-v1 AmputedAnt-v1 BigAnt-v1 ExtendedAnt-v1 
	#python plot_exp.py mom_0.3 SmallInvertedPendulum-v1 BigInvertedPendulum-v1 InvertedPendulum-v1 InvertedDoublePendulum-v1 Hopper-v1 Swimmer-v1 Ant-v1 AmputedAnt-v1 BigAnt-v1 ExtendedAnt-v1 
	#python plot_exp.py mom_0.7 SmallInvertedPendulum-v1 BigInvertedPendulum-v1 InvertedPendulum-v1 InvertedDoublePendulum-v1 Hopper-v1 Swimmer-v1 Ant-v1 AmputedAnt-v1 BigAnt-v1 ExtendedAnt-v1 
	#python plot_exp.py mom_-0.3 SmallInvertedPendulum-v1 BigInvertedPendulum-v1 InvertedPendulum-v1 InvertedDoublePendulum-v1 Hopper-v1 Swimmer-v1 Ant-v1 AmputedAnt-v1 BigAnt-v1 ExtendedAnt-v1 
	#python plot_exp.py post_mom_-0.3 SmallInvertedPendulum-v1 BigInvertedPendulum-v1 InvertedPendulum-v1 InvertedDoublePendulum-v1 Hopper-v1 Swimmer-v1 Ant-v1 AmputedAnt-v1 BigAnt-v1 ExtendedAnt-v1 
	#python plot_exp.py post_mom_0.3 SmallInvertedPendulum-v1 BigInvertedPendulum-v1 InvertedPendulum-v1 InvertedDoublePendulum-v1 Hopper-v1 Swimmer-v1 Ant-v1 AmputedAnt-v1 BigAnt-v1 ExtendedAnt-v1 
	#python plot_exp.py post_mom_0.1 SmallInvertedPendulum-v1 BigInvertedPendulum-v1 InvertedPendulum-v1 InvertedDoublePendulum-v1 Hopper-v1 Swimmer-v1 Ant-v1 AmputedAnt-v1 BigAnt-v1 ExtendedAnt-v1 
	#python plot_exp.py post_mom_0.7 SmallInvertedPendulum-v1 BigInvertedPendulum-v1 InvertedPendulum-v1 InvertedDoublePendulum-v1 Hopper-v1 Swimmer-v1 Ant-v1 AmputedAnt-v1 BigAnt-v1 ExtendedAnt-v1 
	#python plot_exp.py dist_4_vanilla_mom_0.0 SmallInvertedPendulum-v1 BigInvertedPendulum-v1 InvertedPendulum-v1 InvertedDoublePendulum-v1 Hopper-v1 Swimmer-v1 Ant-v1 AmputedAnt-v1 BigAnt-v1 ExtendedAnt-v1 
	python plot_exp.py dist_4_vanilla_mom_0.3 SmallInvertedPendulum-v1 BigInvertedPendulum-v1 InvertedPendulum-v1 InvertedDoublePendulum-v1 Hopper-v1 Swimmer-v1 Ant-v1 AmputedAnt-v1 BigAnt-v1 ExtendedAnt-v1 
	python plot_exp.py dist_8_vanilla_mom_0.0 SmallInvertedPendulum-v1 BigInvertedPendulum-v1 InvertedPendulum-v1 InvertedDoublePendulum-v1 Hopper-v1 Swimmer-v1 Ant-v1 AmputedAnt-v1 BigAnt-v1 ExtendedAnt-v1 
	python plot_exp.py dist_8_vanilla_mom_0.3 SmallInvertedPendulum-v1 BigInvertedPendulum-v1 InvertedPendulum-v1 InvertedDoublePendulum-v1 Hopper-v1 Swimmer-v1 Ant-v1 AmputedAnt-v1 BigAnt-v1 ExtendedAnt-v1 

