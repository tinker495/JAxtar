#!/bin/bash

rm log.txt
PUZZLE=rubikscube

for seed in $(seq 0 300); do
  output=$(python3 main.py qstar -nn -h -p $PUZZLE -w 0.2 --start_state_seed $seed)
  state=$(echo "$output" | grep "Search states: " | awk -F': ' '{print $2}')
  cost=$(echo "$output" | grep "Cost:" | awk '{print $2}')
  time=$(echo "$output" | grep "Time: " | tail -n 1 | awk '{print $2}')

  echo "Seed: $seed, Cost: $cost, Time: $time, State: $state"
  echo "Seed: $seed, Cost: $cost, Time: $time, State: $state" >> log.txt
done
