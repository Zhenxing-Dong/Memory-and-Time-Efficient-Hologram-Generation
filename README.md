## Divide-Conquer-and-Merge: Memory- and Time-Efficient Holographic Displays

> We proposed a divide-conquer-and-merge strategy to address the memory and computational capacity scarcity in ultra-high-definition CGH generation. This algorithm empowers existing CGH frameworks to synthesize higher-definition holograms at a faster speed while maintaining high-fidelity image display quality. Both simulations and experiments were conducted to demonstrate the capabilities of the proposed framework. By integrating our strategy into HoloNet and CCNNs, we achieved significant reductions in GPU memory usage during the training period by 64.3% and 12.9%, respectively. Furthermore, we observed substantial speed improvements in hologram generation, with an acceleration of up to 3x and 2x, respectively. Particularly, we successfully trained and inferred 8K definition holograms on an NVIDIA GeForce RTX 3090 GPU for the first time in simulations. Furthermore, we conducted full-color optical experiments to verify the effectiveness of our method. We believe our strategy can provide a novel approach for memory- and time-efficient holographic displays

<div align=center><img width="90%" src="./related/image.png"/></div>

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
