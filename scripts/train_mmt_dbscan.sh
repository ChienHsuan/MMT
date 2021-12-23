#!/bin/sh
SOURCE="dukemtmc"
TARGET="market1501"         # market1501  dukemtmc  msmt17  lab314  mta
ARCH="msanet_pos"

export PYTHONPATH=$PYTHONPATH:`pwd`
CUDA_VISIBLE_DEVICES=4 \
python examples/mmt_train_dbscan.py -ds ${SOURCE} -dt ${TARGET} -a ${ARCH} \
	--num-instances 4 --lr 0.00035 --iters 400 -b 32 --epochs 40 \
	--soft-ce-weight 0.5 --soft-tri-weight 0.8 --dropout 0 --lambda-value 0 \
	--height 192 --width 96 --features 512 \
	--init-1 /home/lab314/HDD/Hsuan/models/peer_networks/dukemtmc/msanet_pos/1/model.pth.tar-100 \
	--init-2 /home/lab314/HDD/Hsuan/models/peer_networks/dukemtmc/msanet_pos/2/model.pth.tar-100 \
	--data-dir /home/lab314/HDD/Dataset \
	--logs-dir /home/lab314/HDD/Hsuan/MMT/log \
	--eval-step 1 \
	--print-freq 50
