# X-CLIP Mini profile for a single RTX 4060 8 GB (Linux or WSL2).
# The CLI divides --batch_size by --gradient_accumulation_steps, so this
# configuration trains with 4 samples/step and an effective batch of 16.
job_name="xclip_mini_msrvtt_rtx4060"
DATA_PATH="[Your MSRVTT data and videos path]"

mkdir -p log ckpts_dsw/${job_name}

torchrun --standalone --nproc_per_node=1 \
    main_xclip.py --do_train --num_thread_reader=4 \
    --lr 1e-4 --batch_size=16 --batch_size_val=8 \
    --gradient_accumulation_steps=4 \
    --epochs=5 --n_display=10 \
    --train_csv ${DATA_PATH}/MSRVTT/msrvtt_data/MSRVTT_train.9k.csv \
    --val_csv ${DATA_PATH}/MSRVTT/msrvtt_data/MSRVTT_JSFUSION_test.csv \
    --data_path ${DATA_PATH}/MSRVTT/msrvtt_data/MSRVTT_data.json \
    --features_path ${DATA_PATH}/MSRVTT/data/MSRVTT/videos/all \
    --output_dir ckpts_dsw/${job_name} \
    --max_words 24 --max_frames 6 \
    --text_num_hidden_layers 6 --visual_num_hidden_layers 6 \
    --datatype msrvtt --expand_msrvtt_sentences \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 4 --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header meanP \
    --pretrained_clip_name ViT-B/32 2>&1 | tee -a log/${job_name}.log
