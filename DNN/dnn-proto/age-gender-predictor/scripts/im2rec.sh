#!/bin/bash

# execute the im2rec command
$HOME/mxnet/bin/im2rec $1 "" $2 resize=256 encoding='.jpg' quality=100
