set -e -x

MISSING_BIN_FILE=./temp/checkpoints/llavaprotein-2-7b-chat-task-qlora/mm_projector.bin

if [ ! -f $MISSING_BIN_FILE ]
then
  echo "Copying missing file..."
  cp ./temp/llava-v1.5-7b/mm_projector.bin   $MISSING_BIN_FILE
else
  echo "bin file already present [OK]"
fi
