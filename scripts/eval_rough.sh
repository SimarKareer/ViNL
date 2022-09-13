maps=( resources/maps/Adrian0.png resources/maps/Adrian1.png resources/maps/Anaheim1.png resources/maps/Woonsocket0.png resources/maps/map1.png )
export DISPLAY=:1
for map in ${maps[@]}
do
  for ep_id in {0..9}
  do
    python legged_gym/scripts/play.py \
      --task=aliengo_nav \
      --episode-id $ep_id \
      --map $map \
      --seed $ep_id \
      --alt-ckpt weights/Sep11_20-17-29_RoughTerrainDMEnc_model_4450_13.272511272532865.pt \
      --eval-dir evaluation_metrics_rough
  done
done
