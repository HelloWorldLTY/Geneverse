# image_files=(./temp/dataset/images/*)
# first_image="${image_files[0]}"

# python ./llava/eval/run_llava.py \
# --model-path ./temp/checkpoints/llavaallsample-2-7b-chat-task-qlora \
# --model-base ./temp/llava-v1.5-7b \
# --image-file $first_image \
# --query "Is gene FCER1G a marker gene of cell type CD163+ Macrophage? Please summarize your answer in one sentence." \

# #Is gene APOC1 a marker gene of cell type Macrophage? Please summarize your answer in one sentence.
# #Is gene SLC5A6 a marker gene of cell type CRABP2+ Malignant? Please summarize your answer in one sentence.

image_files=(../LLaVA/playground/data/spatial_data_test/*)
first_image="../LLaVA/playground/data/spatial_data_test/showCD4_CD4T.png"

queryi="Is gene CD4 a marker gene of cell type CD4 T? Please summarize your answer in one sentence."

echo $first_image

python ./llava/eval/run_llava.py \
--model-path ./temp/checkpoints/llavafull1-2-7b-chat-task-qlora \
--model-base ./temp/llava-v1.5-7b \
--image-file $first_image \
--query $queryi \

#Is gene APOC1 a marker gene of cell type Macrophage? Please summarize your answer in one sentence.
#Is gene SLC5A6 a marker gene of cell type CRABP2+ Malignant? Please summarize your answer in one sentence.