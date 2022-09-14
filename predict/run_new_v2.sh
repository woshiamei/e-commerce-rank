for i in {1..5};
	do python3 predict_rank_v5.py --per_vids_num=100 --per_insert_num=2000 --jobs_num=10 --jobs_index=$i &>log_$i &
	done
