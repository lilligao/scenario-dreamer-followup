cd /mnt/dataset/perception-aisee/Pubblic_Datasets/nuPlan/nuplan-v0.1/

for f in *.*.*; do
  base="${f%.*}"         # Remove the last part after the final dot
  ext="${base##*.}"      # Extract the original extension
  new="${base}.${ext}"   # Form the new filename
  echo $f
  echo $base
  #aws s3 mv  "$f" "$new"   # Rename
done
