num=6
#dt=`date '+%Y%m%d'`
dt=`date  -d "-1 day" '+%Y%m%d'`
#echo "jobs num: " $num
echo "dt: " $dt
#touch log_new_$dt
for i in {1..6};
	do python3 predict_rank_v5.py --per_vids_num=200 --per_insert_num=5000 --jobs_num=$num --jobs_index=$i --date=$dt &>log_new_$dt &
	done
