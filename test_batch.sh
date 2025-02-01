#!/bin/bash
rm log.txt
rm log_raw.txt

for batch_size in 8192 4096 2048 1024 512 256 128 64 32 16 8 4 2 1; do
  output=$(python3 main.py qstar -b $batch_size)
  state=$(echo "$output" | grep "Search states: " | awk -F': ' '{print $2}')
  cost=$(echo "$output" | grep "Cost: " | awk '{print $2}')
  time=$(echo "$output" | grep "Time: " | tail -n 1 | awk '{print $2}')

  echo "Batch size: $batch_size, Cost: $cost, Time: $time, State: $state"
  echo "Batch size: $batch_size, Cost: $cost, Time: $time, State: $state" >> log.txt
  echo "$output" >> log_raw.txt
  echo "----------------------------------------" >> log_raw.txt
done
