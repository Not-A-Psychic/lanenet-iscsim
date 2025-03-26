$(info USING DATASET=$(DATASET))

DATASET_DIR=/media/jai/Deck/projects/ISC/lanenet/dataset
DATASETS=tusimple culane bdd iscwebots
DATASET=iscwebots

TUSIMPLE_DATA_DIR=$(DATASET_DIR)/tusimple-benchmark
CULANE_DATA_DIR=$(DATASET_DIR)/CULane
BDD_DATA_DIR=$(DATASET_DIR)/bdd/bdd100k

ifeq ($(DATASET), tusimple)
	DATA_DIR=$(TUSIMPLE_DATA_DIR)
	TEST_FILE=$(DATA_DIR)/test_tasks_0627.json
	THICKNESS=5
	IMG_WIDTH=512
	IMG_HEIGHT=256
else ifeq ($(DATASET), culane)
	DATA_DIR=$(CULANE_DATA_DIR)
	TEST_FILE=$(META_DIR)/$(DATASET).json
	THICKNESS=12
	IMG_WIDTH=800
	IMG_HEIGHT=288
else ifeq ($(DATASET), bdd)
	DATA_DIR=$(BDD_DATA_DIR)
	TEST_FILE=$(META_DIR)/$(DATASET).json
	THICKNESS=8
	IMG_WIDTH=800
	IMG_HEIGHT=288
else ifeq ($(DATASET), iscwebots)
	DATA_DIR=$(DATASET_DIR)/iscwebots
	TEST_FILE=$(META_DIR)/iscwebots.json
	THICKNESS=5
	IMG_WIDTH=1280
	IMG_HEIGHT=720
else
	$(error Unknown DATASET $(DATASET))
endif

OUT_DIR=/media/jai/Deck/projects/ISC/lanenet/dataset/output
META_DIR=$(OUT_DIR)/metadata
MODEL_DIR=$(OUT_DIR)/model
TEST_DIR=$(OUT_DIR)/test

GT_FILE=$(DATA_DIR)/test_label.json

BATCH_SIZE?=16
TEST_BATCH_SIZE?=16
LEARNING_RATE?=0.00001
CNN_TYPE?=unet

EXP_NAME=$(DATASET)_$(CNN_TYPE)_b$(BATCH_SIZE)_lr$(LEARNING_RATE)
MODEL_FILE=$(MODEL_DIR)/$(EXP_NAME).pth
PRED_FILE=$(MODEL_FILE:.pth=_predictions.json)
TRAIN_LOG_FILE=$(MODEL_FILE:.pth=_training.log)
TEST_OUT_DIR=$(TEST_DIR)/$(EXP_NAME)

SPLITS=train val

metadata: $(META_DIR)/$(DATASET).json

$(META_DIR)/tusimple.json:
	python src/metadata.py --input_dir $(TUSIMPLE_DATA_DIR) \
		--dataset tusimple \
		--output_file $@

$(META_DIR)/culane.json:
	python src/metadata.py --input_dir $(CULANE_DATA_DIR) \
		--dataset culane \
		--output_file $@

$(META_DIR)/bdd.json:
	python src/metadata.py --input_dir $(BDD_DATA_DIR) \
		--dataset bdd \
		--output_file $@

$(META_DIR)/iscwebots.json:
	python src/metadata.py --input_dir $(DATASET_DIR)/iscwebots \
		--dataset iscwebots \
		--output_file $@

BIN_DIR=$(OUT_DIR)/bin_images
INS_DIR=$(OUT_DIR)/ins_images

generate_label_images:
	python src/gen_seg_images.py $(META_DIR)/tusimple.json $(DATA_DIR) \
		--bin_dir $(BIN_DIR) \
		--ins_dir $(INS_DIR) \
		--splits train val \
		--thickness $(THICKNESS)

train: $(MODEL_FILE)
$(MODEL_FILE): $(META_DIR)/$(DATASET).json 
	python src/train.py $^ $@ \
		--image_dir $(DATA_DIR) \
		--batch_size $(BATCH_SIZE) \
		--num_workers 8 \
		--cnn_type $(CNN_TYPE) \
		--embed_dim 4 \
		--dataset $(DATASET) \
		--width $(IMG_WIDTH) \
		--height $(IMG_HEIGHT) \
		--thickness $(THICKNESS) \
		2>&1 | tee $(TRAIN_LOG_FILE)

test: $(PRED_FILE)
$(PRED_FILE): $(MODEL_FILE) $(TEST_FILE) 
	python src/test.py $< \
		--output_file $@ \
		--meta_file $(word 2, $^) \
		--image_dir $(DATA_DIR) \
		--output_dir $(TEST_OUT_DIR) \
		--loader_type $(DATASET)test \
		--num_workers 8 \
		--batch_size $(TEST_BATCH_SIZE)

SHELL=/bin/bash 
eval_tusimple: $(PRED_FILE) $(GT_FILE) 
	source activate py2 && \
		python tusimple-benchmark/evaluate/lane.py $^ && \
	source deactivate

demo_tusimple: $(MODEL_FILE) $(META_DIR)/tusimple.json
	python src/test.py $< \
		--meta_file $(word 2, $^) \
		--image_dir $(DATA_DIR) \
		--output_dir $(OUT_DIR)/demo_tusimple \
		--loader_type meta \
		--batch_size 1 --show_demo

