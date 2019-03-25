# Super-resolution
# test command 
python test.py --cuda --GPUnum 1 --input_downsample 1 --chop_forward True --output Results1/ --GPUnum 2 --model_name model/model_epoch_9_loss723.9285278320_lr0.0001000000.pth

python test.py --cuda --GPUnum 1 --input_downsample 1 --chop_forward True --output Results2/ --GPUnum 2 --model_name model/model_epoch_36_loss618.6967163086_lr0.0001000000.pth
# Results
The output of different model is in the path 'Input/Test_LR/Results1/' or 'Input/Test_LR/Results2/' which depends on the args.output.
The output of bicubic is in the path 'Input/Test_LR/Results/bicubic/' 
