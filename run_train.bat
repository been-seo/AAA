G:  
cd G:\Air\AAA  
python -u -m ai.world_model.trainer --data-dir data/recordings --epochs 50 --batch-size 16384 --lr 3e-4 --save-dir models/world_model_v2 --log-db models/world_model_v2/train_log.db >models/world_model_v2/train_stdout.log 2>&1  
