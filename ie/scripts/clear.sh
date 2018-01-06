#!/bin/bash

[ -d ../checkpoints ] && rm -r ../checkpoints
[ -d ../examples ] && rm -r ../examples
[ -d ../results ] && rm -r ../results
[ -d ../graphs ] && rm -r ../graphs
[ -f ../nohup.out ] && rm ../nohup.out
