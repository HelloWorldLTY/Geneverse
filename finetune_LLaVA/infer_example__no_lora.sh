image_files=(./temp/dataset/images/*)
first_image="${image_files[0]}"

echo "Infer WITHOUT the lora fine-tuning layer"
python ./llava/eval/run_llava.py \
--model-path ./temp/llava-v1.5-7b \
--image-file $first_image \
--query "Is gene showKRT7 a marker gene of cell type ACTA2+ Myoepi?"
