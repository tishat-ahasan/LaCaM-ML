#!/bin/bash


agent=50
while [ $agent -le 300 ]; do
	echo "Agent: $agent"
	seed=1
	while [ $seed -le 1000 ]; do
	    build/main -v 1 -m assets/random-32-32-10.map -N $agent -s $seed
	    seed=$((seed + 1))
	done
	agent=$((agent + 50))
done

