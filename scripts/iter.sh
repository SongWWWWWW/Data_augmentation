# /bin/bash

# python iter.py --task_name "prompt2_3.1_7_3_0.8" --sample_rate 0.8
# python iter.py --task_name "prompt2_3.1_7_3_0.0" --sample_rate 0.0
# python iter.py --task_name "strategy1_3.1_7_3_0.0" --sample_rate 0.0 --data_generate_strategy "strategy1"
# python iter.py --task_name "strategy1_3.1_7_3_0.0_generate1_test" --sample_rate 0.0 --data_generate_strategy "strategy1" --generate_num 1
# python iter.py --task_name "strategy1_3.1_7_3_0.0_generate5_augmetation5" \
#     --sample_rate 0.0 \
#     --data_generate_strategy "strategy1" \
#     --generate_num 5\
#     --augmentation_rate 0.1\
#     --augmentation_num 5

# python iter.py --task_name "strategy1_3.1_7_3_0.0_generate5_augmetation10" \
#     --sample_rate 0.0 \
#     --data_generate_strategy "strategy1" \
#     --generate_num 5\
#     --augmentation_rate 0.1\
#     --augmentation_num 5

# python iter.py --task_name "strategy1_3.1_7_3_0.0_generate5_augmetation1_rate3" \
#     --sample_rate 0.0 \
#     --data_generate_strategy "strategy1" \
#     --generate_num 5\
#     --augmentation_rate 0.3\
#     --augmentation_num 1

# python iter.py --task_name "strategy1_3.1_7_3_0.0_generate5_augmetation1_rate2" \
#     --sample_rate 0.0 \
#     --data_generate_strategy "strategy1" \
#     --generate_num 5\
#     --augmentation_rate 0.2\
#     --augmentation_num 1

python iter.py --task_name "strategy1_3.1_7_3_0.0_generate5_augmetation1_rate3_new_prompt" \
    --sample_rate 0.0 \
    --data_generate_strategy "strategy1" \
    --generate_num 5\
    --augmentation_rate 0.3\
    --augmentation_num 1
python iter.py --task_name "strategy1_3.1_7_3_0.0_generate5_augmetation1_rate2_new_prompt" \
    --sample_rate 0.0 \
    --data_generate_strategy "strategy1" \
    --generate_num 5\
    --augmentation_rate 0.2\
    --augmentation_num 1