#!/usr/bin/bash

function run() {
	cd /root/reco/rank/predict 
	lock_file='./.lock_old'

	done_file=./done_file_old
	last_time=`head -1 $done_file`
	lt=`date -d "$last_time" +%Y%m%d`
	echo "last date: " $lt

	dt=`date '+%Y%m%d'`
	dt2=`date +%Y%m%d,%H`

	if [ $dt -eq $lt ];then
		echo "$dt 已经执行过，忽略本次"
		return
	else
		echo "未执行 $dt"
	fi

	if [ -f $lock_file ];then
	  echo "last task is running; drop current"
	  return
	fi
	num=4
	echo "jobs num: " $num
	echo "dt2: " $dt2
	touch $lock_file 
	python3 predict_rank_v3.py --per_vids_num=500 --per_insert_num=5000 --jobs_num=$num --date=$dt &>log_old_$dt2 
	rm $lock_file
	# done_file
	#echo "$dt" > $done_file
	# python3 predict_rank_v4.py --per_vids_num=500 --per_insert_num=2000 --jobs_num=6 &>predict_new_20220816.log &
	# python3 predict_rank_v5.py --per_vids_num=100 --per_insert_num=1000 --jobs_num=10 --jobs_index=1
}

$@
