#!/bin/bash

#SBATCH -p general	# partition (queue)
#SBATCH -N 1		# number of nodes
#SBATCH -n 8		# number of cores
#SBATCH -t 0-2:00	# time (D-HH:MM)
#SBATCH -o slurm.coarse.%N.%j.out # STDOUT
#SBATCH -e slurm.coarse.%N.%j.err # STDERR
#SBATCH --mail-type=END,FAIL

time ./sobel_coarse space_station_hires.jpg

