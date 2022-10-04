python ./code/train_ccnet_3d_v1.py --dataset_name LA --model ccnet3d_v1 --exp CCNET --labelnum 16 --gpu 0 --temperature 0.1 --max_iteration 10000  && \
python ./code/train_ccnet_3d_v1.py --dataset_name LA --model ccnet3d_v1 --exp CCNET --labelnum 8 --gpu 0 --temperature 0.1 --max_iteration 10000  && \

python ./code/test.py --dataset_name LA --model ccnet3d_v1 --exp CCNET --labelnum 16 --gpu 0 && \
python ./code/test.py --dataset_name LA --model ccnet3d_v1 --exp CCNET --labelnum 8 --gpu 0