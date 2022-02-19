#!/bin/bash

cd /home/ec2-user/user-simulation


aws s3 cp s3://user-simulation-t5-models/act-sat_no-alt/MWOZ/ simpleT5/act-sat_no-alt/MWOZ/ --recursive
aws s3 cp s3://user-simulation-t5-models/act-sat_no-alt/SGD/ simpleT5/act-sat_no-alt/SGD/ --recursive
aws s3 cp s3://user-simulation-t5-models/act-sat_no-alt/CCPE/ simpleT5/act-sat_no-alt/CCPE/ --recursive

aws s3 cp s3://user-simulation-t5-models/act-sat-utt_no-alt/MWOZ/ simpleT5/act-sat-utt_no-alt/MWOZ/ --recursive
aws s3 cp s3://user-simulation-t5-models/act-sat-utt_no-alt/SGD/ simpleT5/act-sat-utt_no-alt/SGD/ --recursive
aws s3 cp s3://user-simulation-t5-models/act-sat-utt_no-alt/CCPE/ simpleT5/act-sat-utt_no-alt/CCPE/ --recursive
