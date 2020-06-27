#!/bin/sh

#Take auxiliary loss trained network and truncates down into embedding network
#This script takes three arguments
# $1 is the location of the learned network files, e.g., ../../pretrained_nets/auxloss/
# $2 is the filename search phrase for all files that are to be distilled, e.g. allbutrecon_64.params
# $3 is the number of embedding dimensions, must match actual network params
#usage example:
# bash strip_to_embedding_networks.sh ../../pretrained_networks/auxloss/ allbutrecon_64.params 64
for i in `ls $1*.params | grep $2`
do
  orig=$i
  ending="_stripped.params"
  new=$i$ending
  echo "distilling $orig --> $new"
  echo `python ../transfer.py $orig $new $3`
done
