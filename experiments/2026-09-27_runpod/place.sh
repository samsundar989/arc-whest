#!/bin/bash
# Place an uploaded batch on the pod and raise its flag. Usage: place.sh w100|labels|bench|full
set -e
A=/workspace/arc; I=/workspace/in; mkdir -p $A/cache/twin
case $1 in
w100)
  cd $A/cache/twin
  truncate -s $((1000*16*1024*1024*4)) full_w.f32
  dd if=$I/stage/full_w_0_99.f32 of=full_w.f32 bs=64M conv=notrunc status=none && rm $I/stage/full_w_0_99.f32
  mv $I/stage/full_truth.npy $I/stage/full_meta.json $I/stage/mini_truth.npy $I/stage/mini_meta.json .
  mv $I/mini_w.f32 .
  cd $A && python tools/twin/export_parquet.py --verify /workspace/arcwhest_sha.json || true
  touch /workspace/READY_W100 ;;
labels) rm -rf $A/cache/twin_labels && mv $I/twin_labels $A/cache/ && ls $A/cache/twin_labels | wc -l && touch /workspace/READY_LABELS ;;
bench)
  mkdir -p $A/cache/hf/arc-whestbench-p2-full1000-N1e9 $A/cache/hf/whest-p2-bakev2-bench-supp
  rm -rf $A/cache/hf/arc-whestbench-p2-full1000-N1e9/moments && mv $I/moments $A/cache/hf/arc-whestbench-p2-full1000-N1e9/
  rm -rf $A/cache/hf/whest-p2-bakev2-bench-supp/s00 && mv $I/s00 $A/cache/hf/whest-p2-bakev2-bench-supp/
  touch /workspace/READY_BENCH ;;
full)
  mv $I/full_w.f32 $A/cache/twin/full_w.f32 && mv $I/full_meta_all.json $A/cache/twin/full_meta.json
  cd $A && python tools/twin/export_parquet.py --verify /workspace/arcwhest_sha.json && touch /workspace/READY_FULL ;;
esac
echo PLACED $1
