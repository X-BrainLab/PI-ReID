PROJECT_ROOT_DIR=/root/PI-ReID
DATASETS_ROOT_DIR=$PROJECT_ROOT_DIR/datasets
PRETRAINED_PATH=$PROJECT_ROOT_DIR/pretrained/prw/resnet50_model_120.pth
OUTPUT=$PROJECT_ROOT_DIR/output/prw
Pre_Index_DIR=$PROJECT_ROOT_DIR/pre_index_dir/prw_pre_index.json

python tools/pre_selection.py --config_file='configs/softmax_triplet_ft.yml' MODEL.DEVICE_ID "('3')" DATASETS.NAMES "('prw')" \
DATASETS.ROOT_DIR $DATASETS_ROOT_DIR MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT $PRETRAINED_PATH \
OUTPUT_DIR $OUTPUT \
Pre_Index_DIR $Pre_Index_DIR

python3 tools/train.py --config_file='configs/softmax_triplet_ft.yml' MODEL.DEVICE_ID "('3')" DATASETS.NAMES "('prw')" DATASETS.ROOT_DIR $DATASETS_ROOT_DIR \
OUTPUT_DIR $OUTPUT SOLVER.BASE_LR 0.00035 TEST.PAIR "no" SOLVER.IMS_PER_BATCH 64 \
MODEL.WHOLE_MODEL_TRAIN "no" MODEL.PYRAMID "s2" MODEL.SIA_REG "yes" MODEL.GAMMA 1.0 SOLVER.MARGIN 0.1 MODEL.BETA 0.5 DATASETS.TRAIN_ANNO 1 SOLVER.EVAL_PERIOD 15 SOLVER.MAX_EPOCHS 50 \
MODEL.PRETRAIN_PATH $PRETRAINED_PATH \
Pre_Index_DIR $Pre_Index_DIR