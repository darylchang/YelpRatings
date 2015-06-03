#!/bin/bash

#verbose
set -x
###################
# Update items below for each train/test
###################

# training params
epochs=97
step=1e-2
wvecDim=30

# for RNN2 only, otherwise doesnt matter
middleDim=45

model="RNTN" #either RNN, RNN2, RNN3, RNTN, or DCNN


######################################################## 
# Probably a good idea to let items below here be
########################################################
if [ "$model" == "RNN2" ]; then
    outfile="models/${model}_wvecDim_${wvecDim}_middleDim_${middleDim}_step_${step}_2.bin"
else
    outfile="models/${model}_wvecDim_${wvecDim}_step_${step}_2.bin"
fi


echo $outfile

#command to run from random initialization
# python runNNet.py --step $step --epochs $epochs --outFile $outfile \
#                 --middleDim $middleDim --outputDim 5 --wvecDim $wvecDim --model $model 

#command to run from pre-trained and train on Yelp data?
infile="models/${model}_wvecDim_${wvecDim}_step_${step}_2.bin"
outfile="models/${model}_wvecDim_${wvecDim}_step_${step}_2_Yelp.bin"
python runYelp.py --step $step --epochs $epochs --infile $infile --outFile $outfile2 \
				--middleDim $middleDim --outputDim 5 --wvecDim $wvecDim --model $model