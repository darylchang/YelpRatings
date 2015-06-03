#!/bin/bash

# verbose
set -x


infile="models/RNTN_wvecDim_30_step_1e-2_2_Yelp.bin" # the pickled neural network
model="RNTN" # the neural network type

echo $infile

# test the model on test data
#python runNNet.py --inFile $infile --test --data "test" --model $model

# test the model on dev data
# python runNNet.py --inFile $infile --test --data "dev" --model $model

# test the model on training data
#python runNNet.py --inFile $infile --test --data "train" --model $model


# ========== TEST ON YELP DATA =============

# test the model on Yelp test data
#python runYelp.py --inFile $infile --test --data "test" --model $model

# test the model on Yelp dev data
python runYelp.py --inFile $infile --test --data "dev" --model $model

# test the model on Yelp training data
#python runYelp.py --inFile $infile --test --data "train" --model $model



