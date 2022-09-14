dt=`date '+%Y%m%d'`
python3 predict_rank_v3.py --per_vids_num=500 --per_insert_num=2000 --jobs_num=4 --date=$dt &>predict_user_$dt.log 
# python3 predict_rank_v4.py --per_vids_num=500 --per_insert_num=2000 --jobs_num=6 &>predict_new_20220816.log &
# python3 predict_rank_v5.py --per_vids_num=100 --per_insert_num=1000 --jobs_num=10 --jobs_index=1
