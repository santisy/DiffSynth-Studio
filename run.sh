JOB_NAME=lora-finetune-1007-filtered-02
rjob delete $JOB_NAME
rjob submit \
	--name=$JOB_NAME \
	--gpu=8 \
	--memory=900000 \
	--cpu=80 \
	--charged-group=idc2_gpu \
	--private-machine=group \
	--mount=gpfs://gpfs1/yangdingdong:/mnt/shared-storage-user/yangdingdong \
	--image=registry.h.pjlab.org.cn/ailab-idc2-idc2_gpu/jianglihan:base -- \
	bash -exc /mnt/shared-storage-user/yangdingdong/code/DiffSynth-Studio/run_exp.sh
