#!/bin/bash

# List of file names (without .mp4)
cases=(
  "case_1603" "case_0888" "case_0152" "case_0342"
  "case_0429" "case_0984" "case_0735" "case_1119"
  "case_1905" "case_1884" "case_0183" "case_1721"
  "case_1315" "case_1555" "case_1210" "case_0043"
  "case_1054" "case_0998" "case_0776" "case_1492"
)

# Source and target directories
source_dir="$HOME/CS/DeepLearning/APTOS_2025/dataset/dataset/APTOS_train-val/aptos_videos"
target_dir="$HOME/CS/DeepLearning/APTOS_2025/repos/aptos2025/dataset/videos"

# Loop through and move each .mp4 file
for case in "${cases[@]}"; do
  src_file="$source_dir/$case.mp4"
  if [[ -f "$src_file" ]]; then
    mv "$src_file" "$target_dir/"
    echo "Moved $case.mp4 to $target_dir"
  else
    echo "File $case.mp4 not found in $source_dir"
  fi
done
