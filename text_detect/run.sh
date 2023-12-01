CUDA_VISIBLE_DEVICES=0 python run.py \
--Transformation TPS \
--FeatureExtraction ResNet \
--SequenceModeling BiLSTM \
--Prediction Attn \
--image_folder demo_image/ \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-Attn.pth \
# --frame 1 \
# --use_dl \
# --use_obs \
# --show \