## Divide-Conquer-and-Merge: Memory- and Time-Efficient Holographic Displays

<div align=center><img width="80%" src="./related/image.png"/></div>
## Setup 
Install the required packages using conda with the provided [environment.yaml](https://github.com/Zhenxing-Dong/Memory-and-Time-Efficient-Hologram-Generation/blob/main/environment.yaml) file.

## Train
    python train_holonet.py --channel <CHANNEL_OF_LIGHT> --run_id <EXPERIMENT_NAME> --num_epochs <EPOCHS_NUM> --lr <LEARNING_RATE> --loss_fun <LOSS_FUNCTION> --perfect_prop_model True --method <METHOD> --scale_factor <SCALE_FACTOR_OF_PIXEL_SHUFFLE> --res <IMAGE_DEFINITION>
## Test
    python main.py --channel <CHANNEL_OF_LIGHT> --method <METHOD> --res <IMAGE_DEFINITION> --scale_factor <SCALE_FACTOR_OF_PIXEL_SHUFFLE> --checkpoint <NAME_OF_TRAINED_CHECKPOINT>
## Acknowledgement
The codes are built on [neural holography](https://github.com/computational-imaging/neural-holography). We sincerely appreciate the authors for sharing their codes.

## Contact
If you have any questions, please do not hesitate to contact [d_zhenxing@sjtu.edu.cn](d_zhenxing@sjtu.edu.cn).
