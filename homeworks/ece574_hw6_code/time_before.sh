#!/bin/bash

#SBATCH -p general	# partition (queue)
#SBATCH -t 0-2:00	# time (D-HH:MM)
#SBATCH -o slurm.before.%N.%j.out # STDOUT
#SBATCH -e slurm.before.%N.%j.err # STDERR

time ./sobel_before space_station_hires.jpg
