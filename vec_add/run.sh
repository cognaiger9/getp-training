#!/bin/bash

srun --nodes=1 --exclusive ./main $@
