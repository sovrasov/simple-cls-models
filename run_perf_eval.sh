declare -a models=("resnet18" "mobilenetv3_large" "efficient_net_v2_s" "efficient_net_b0")
declare -a devices=("xpu" "cpu" "cuda")

for model in "${models[@]}"
do
    for dev in "${devices[@]}"
    do
        command="train.py --config ./configs/$model.py --device $dev --output_dir ./$model-$dev"
        #python3 $command | tee -a ./$model-$dev.log
        echo $command
    done
done