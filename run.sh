#!/bin/bash
for dataset in "CIFAR10" "MNIST" "Fashion-MNIST";
do
  for bits in {12..48..12}
  do
    th main.lua -bits $bits -dataset $dataset
  done
done

exit 1
