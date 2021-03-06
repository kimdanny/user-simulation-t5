#!/bin/bash

cd /home/ec2-user/user-simulation

aws s3 cp simpleT5/act-sat_no-alt/SGD/ s3://user-simulation-t5-models/act-sat_no-alt/SGD/ --recursive
aws s3 cp simpleT5/act-sat_no-alt/MWOZ/ s3://user-simulation-t5-models/act-sat_no-alt/MWOZ/ --recursive
aws s3 cp simpleT5/act-sat_no-alt/CCPE/ s3://user-simulation-t5-models/act-sat_no-alt/CCPE/ --recursive

aws s3 cp simpleT5/act-sat-utt_no-alt/SGD/ s3://user-simulation-t5-models/act-sat-utt_no-alt/SGD/ --recursive
aws s3 cp simpleT5/act-sat-utt_no-alt/MWOZ/ s3://user-simulation-t5-models/act-sat-utt_no-alt/MWOZ/ --recursive
aws s3 cp simpleT5/act-sat-utt_no-alt/CCPE/ s3://user-simulation-t5-models/act-sat-utt_no-alt/CCPE/ --recursive

aws s3 cp simpleT5/utt_no-alt/ s3://user-simulation-t5-models/utt_no-alt/ --recursive
