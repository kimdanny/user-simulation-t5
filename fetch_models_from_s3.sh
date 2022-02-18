#!/bin/bash

cd /home/ec2-user/user-simulation

echo "syntax: bash fetch_models_from_s3.sh <DATASET>"

aws s3 cp s3://user-simulation-t5-models/act-sat_no-alt/$1/ simpleT5/act-sat_no-alt/$1/ --recursive
