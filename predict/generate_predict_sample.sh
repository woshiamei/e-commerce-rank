echo "generate predict sample starting!"
dt=`date '+%Y%m%d'`
echo $dt
#dt2=`date  -d "-10 day" '+%Y%m%d'`
#echo $dt2
#cd /root/reco/rank/train && python3 nn_train_v7_1.py --start_date=$dt --date_num=12 >nn_train_v7_1.log &
cd /root/reco/rank/predict && python3 generate_predict_sample.py --date=$dt --overwirte=False >generate_predict_sample.log
#--overwirte='True'
#cd /root/reco/rank/train && python3 nn_train_v4.py --start_date='20220801' --date_num=1 >nn_train_v3.log

echo 'done!'