The code of Glance to Count: Learning to Rank with Anchors for Weakly-supervised Crowd Counting.



Train code is 

```sh
python3 train.py --dataset_name ShanghaiTechA --eval_mode ShanghaiTechA --end_step 1000 --eval_step 1 --experiment-ID 1 --gpu_id 1
```



Eval code is 

```sh
python3 train_baseline.py --dataset_name ShanghaiTechA_Combine --compare_loss_mode --eval_mode Rank --end_step 1000 --eval_step 1 --experiment-ID eval_1 --gpu_id 1 --lambda_reg 0.2 --lr 0.000005 --save_all_weights
```

