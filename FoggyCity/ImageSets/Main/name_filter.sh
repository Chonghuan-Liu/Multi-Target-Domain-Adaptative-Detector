#!/bin/bash

FILENAME="trainval_full.txt"
DETNAME="trainval.txt"

cat $FILENAME | while read line
do
	if [ ${line:(-4)} == "0.02" ] || [ ${line:0:6} == "source" ];then
		echo $line >> $DETNAME
	fi
done
