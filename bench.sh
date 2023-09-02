#!/bin/bash

if [ "$#" -ne 2 ] && [ "$#" -ne 3 ]; then
  echo "Usage: ./bench.sh <package> <type> [<max_size>]"
  exit -1
fi

package=$1
dir=data/$2
size=${3:--1}

echo Parsing $dir

NUM_CORES=$(nproc --all)
MAX_JOBS=1 # $(($NUM_CORES * 2))
SUB='.qsim'

function run_process() {
  python3 src/bench.py -p $package -f $dir/$1 -s $size
  # sleep 0
}

for file in $(ls $dir); do
  IN=$file
  arrIN=(${IN//_/ })

  # Quantum circuit.
  if [[ "$IN" == *"$SUB"* ]]; then
    run_process $file
  else
    # Synthetic tensor network.
    if [ $((${arrIN[0]} + 0)) -le $size ] || [ $size -eq -1 ]; then
      run_process $file
      # &
      # while [ `jobs | wc -l` -ge $MAX_JOBS ]
      # do
      #   sleep 0
      # done
    fi
  fi
done;
wait