cd /root/reco/rank/predict
# old user
processID=(`ps -ef | grep predict_rank_v3 |grep -v "grep" |awk '{print $2}'`)
#processID=(`ps -ef | grep mem |grep -v "grep" |awk '{print $2}'`)
echo $processID
lock_file='./.lock_old'
if [ -z "$processID" ]
then
	echo "不存在该进程！predict_rank_v3"
	if [ -f $lock_file ];then
		echo "last task lock file need delete!"
		rm $lock_file
	  	return
	fi
else
	echo "存在该进程！"
	for pid in ${processID[@]}
	do
		run_time=`ps -o lstart,etime -p $pid | tail -1 | awk '{print $6}' | awk -F':' '{print $1}'`
		#echo ${#run_time}
		echo "pid run_time: " $pid $run_time
		if [ ${#run_time} -gt 3 ]
		then 
			echo "kill -9 " $pid
			kill -9 $pid
		else
			echo "wait pid: " $pid
		fi
	done
fi

#new user
processID=(`ps -ef | grep predict_rank_v5 |grep -v "grep" |awk '{print $2}'`)
#processID=(`ps -ef | grep mem |grep -v "grep" |awk '{print $2}'`)
echo $processID
lock_file='./.lock_new'
if [ -z "$processID" ]
then
	echo "不存在该进程！predict_rank_v5"
	if [ -f $lock_file ];then
		echo "last task lock file need delete!"
		rm $lock_file
	  	return
	fi
else
	echo "存在该进程！"
	for pid in ${processID[@]}
	do
		run_time=`ps -o lstart,etime -p $pid | tail -1 | awk '{print $6}' | awk -F':' '{print $1}'`
		#echo ${#run_time}
		echo "pid run_time: " $pid $run_time
		if [ ${#run_time} -gt 3 ]
		then 
			echo "kill -9 " $pid
			kill -9 $pid
		else
			echo "wait pid: " $pid
		fi
	done
fi
