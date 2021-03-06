#!/bin/bash

#SBATCH -p general	# partition (queue)
#SBATCH -t 0-0:10	# time (D-HH:MM)
#SBATCH -o slurm.coarse.%N.%j.out # STDOUT
#SBATCH -e slurm.coarse.%N.%j.err # STDERR

srun ./sobel_fine space_station_hires.jpg

