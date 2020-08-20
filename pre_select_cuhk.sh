
python3 tools/pre_selection.py --config_file='configs/softmax_triplet_ftc.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('cuhk')" \
DATASETS.ROOT_DIR "('/root/person_search/dataset')" MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('/root/person_search/trained/strong_baseline/cuhk_all_trick_1/resnet50_model_120.pth')"  \
OUTPUT_DIR "('/root/person_search/trained/multi_person/cuhk_all_trick_1')"
