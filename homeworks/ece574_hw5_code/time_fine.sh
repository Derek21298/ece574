#!/bin/bash

#SBATCH -p general	# partition (queue)
#SBATCH -N 1		# number of nodes
#SBATCH -n 8		# number of cores
#SBATCH -t 0-2:00	# time (D-HH:MM)
#SBATCH -o slurm.fine.%N.%j.out # STDOUT
#SBATCH -e slurm.fine.%N.%j.err # STDERR
#SBATCH --mail-type=END,FAIL

export OMP_NUM_THREADS=1
time ./sobel_fine space_station_hires.jpg

