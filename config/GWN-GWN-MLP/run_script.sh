#!/bin/bash
source activate pt_11
cd ./SINet/

model="GWN-GWN-MLP"

gpunum=$(nvidia-smi -L | wc -l)
nps=$((gpunum*6))

threadnum=4
export MKL_NUM_THREADS=${threadnum}
export NUMEXPR_NUM_THREADS=${threadnum}
export OMP_NUM_THREADS=${threadnum}

cfg=`ls -1v ./SINet/config/$model/script/script_GWN-GWN-MLP_* | shuf`
echo $cfg

gpu=0
for _cfg in ${cfg[@]} ; do
  echo $_cfg
  for _expid in {0..9} ; do
    _nps=`ps -aux | grep $model | awk -F "expid" '{print $2}' | sort | uniq | wc -l`

    if [ $_nps -ge $nps ] ; then
        printf "[Process Number Report] [Over] P > P_MAX [%s>%s]\n" $_nps $nps
        sleep 20s
    else
        printf "[Process Number Report] [Under] P < P_MAX [%s<%s]\n" $_nps $nps

        export CUDA_VISIBLE_DEVICES=$(expr $gpu % $gpunum)
        printf "Export %s / %s GPU\n" $(expr $gpu % $gpunum) $gpunum

        printf "Run %s \n" $_cfg
        python demo_gwn_gwn_mlp.py --expid ${_expid} --config ${_cfg} &\

        gpu=$(expr $gpu + 1)
        sleep 1s
    fi
  done
done
