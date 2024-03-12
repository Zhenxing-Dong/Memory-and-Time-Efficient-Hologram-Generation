## Divide-Conquer-and-Merge: Memory- and Time-Efficient Holographic Displays [IEEE VR 2024]

## Training
    python train_holonet.py --channel <CHANNEL_OF_LIGHT> --run_id <EXPERIMENT_NAME> --num_epochs <EPOCHS_NUM> --lr <LEARNING_RATE> --loss_fun <LOSS_FUNCTION> --perfect_prop_model True --method <METHOD> --scale_factor <SCALE_FACTOR_OF_PIXEL_SHUFFLE> --res <IMAGE_DEFINITION>
## Testing
    python main.py --channel <CHANNEL_OF_LIGHT> --method <METHOD> --res <IMAGE_DEFINITION> --scale_factor <SCALE_FACTOR_OF_PIXEL_SHUFFLE> --checkpoint <NAME_OF_TRAINED_CHECKPOINT>
## Acknowledgement
The codes are built on [neural holography](https://github.com/computational-imaging/neural-holography). We sincerely appreciate the authors for sharing their codes.
