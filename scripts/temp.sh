# /bin/bash
python eval.py\
    --model_root_path './results/baseline'\
    --test_path '../vast/raw_test_all_onecol.csv'\
    --output_error_question False\
    --save_log True\
    --sign "test"