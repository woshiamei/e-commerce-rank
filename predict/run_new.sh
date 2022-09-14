#!/usr/bin/bash

function run() {
	cd /root/reco/rank/predict 
	lock_file='./.lock_new'
	done_file=./done_file_new
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
	touch $lock_file
	
	num=9
	num_all=$[$num+1]
	#dt=`date  -d "-1 day" '+%Y%m%d'`
	echo "jobs num: " $num_all
	echo "dt2: " $dt2
	#touch log_new_$dt
	#for i in {1..6};
	for i in `eval echo {1..$num}`
		do 
			echo $i
			python3 predict_rank_v5.py --per_vids_num=500 --per_insert_num=5000 --jobs_num=$num_all --jobs_index=$i --date=$dt &>log_new_$dt2 &
			sleep 1m
		done
	echo $num_all
	python3 predict_rank_v5.py --per_vids_num=500 --per_insert_num=5000 --jobs_num=$num_all --jobs_index=$num_all --date=$dt &>log_new_$dt2
	rm $lock_file
	# done_file
	# echo "$dt" > $done_file
}

$@
