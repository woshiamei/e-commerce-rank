#source /root/gaolei/venv_gl.sh
echo "get raw sample 任务执行"

cd /root/reco/rank/get_data && python3 get_data.py &>get_data.log
