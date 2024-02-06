#!/bin/bash

: ${NODES:=2}

salloc -N $NODES --exclusive                         \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  ./main -v -n 3 4 4 4 -p $@
