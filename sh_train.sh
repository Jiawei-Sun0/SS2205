#!/bin/bash
# for `seq 10` {} 
for i in {0..10}
do
    python training_segmentation.py training/ validation/ output/
done