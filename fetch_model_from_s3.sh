#!/bin/bash

cd /home/ec2-user/user-simulation

echo "syntax: bash fetch_model_from_s3.sh <TASK> <DATASET>"

aws s3 cp s3://user-simulation-t5-models/$1/$2/ simpleT5/$1/$2/ --recursive
