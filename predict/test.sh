#cur_dateTime=`date +%Y-%m-%d,%H`#echo $cur_dateTime
#echo $cur_dateTime
done_file=./done_file_old
dt=`date '+%Y%m%d'`
echo $dt
last_time=`head -1 $done_file`
echo $last_time
dt2=`date -d "$last_time" +%Y%m%d`
echo $dt2
if [ $dt -eq $dt2 ];then
	echo "$dt 已经执行过，忽略本次"
else
	echo "未执行 $dt"
	# return
fi
echo "$dt" > $done_file
