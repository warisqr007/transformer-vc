. ./path.sh || exit 1;
export CUDA_VISIBLE_DEVICES=0

stage=1
stop_stage=1
# config=/mnt/data2/bhanu/code-bases/ppg-vc/conf/transformer_vc_ppg2mel_diff.yaml
# model_file=/mnt/data2/bhanu/code-bases/ppg-vc/ckpt/convpos_5split/step_654000.pth
# config=/mnt/data2/bhanu/code-bases/ppg-vc/conf/transformer_vc_ppg2mel_cmu.yaml
# model_file=/mnt/data2/bhanu/code-bases/ppg-vc/ckpt/transformer_vc_ppg2mel_cmu_seed0/best_loss_step_142000.pth
# config=/mnt/data2/bhanu/code-bases/ppg-vc/conf/transformer_vc_ppg2mel_outspkdloss.yaml
# model_file=/mnt/data2/bhanu/code-bases/ppg-vc/ckpt/convpos_spkloss/step_253000.pth

# config=/mnt/data2/bhanu/code-bases/ppg-vc/conf/transformer_vc_ppg2mel_outspkdloss_inp_conct.yaml
# model_file=/mnt/data2/bhanu/code-bases/ppg-vc/ckpt/convpos_spkloss2_8020/best_loss_step_587000.pth

# config=/mnt/data2/bhanu/code-bases/ppg-vc/conf/transformer_vc_ppg2mel_spkdloss_melsIdirect.yaml
# model_file=/mnt/data2/bhanu/code-bases/ppg-vc/ckpt/convpos_spkloss_melin/step_713000.pth

config=/mnt/data2/bhanu/code-bases/ppg-vc/conf/transformer_vc_ppg2mel_outspkdloss_inp_conct.yaml
model_file=/mnt/data2/bhanu/code-bases/ppg-vc/ckpt/convpos_spkloss_inputconc_new/best_loss_step_1000000.pth

# config=/mnt/data2/bhanu/code-bases/ppg-vc/conf/transformer_vc_ppg2mel_outspkdloss_attn_conct.yaml
# model_file=/mnt/data2/bhanu/code-bases/ppg-vc/ckpt/convpos_spkloss_attnconc_new/best_loss_step_700000.pth

 


src_wav_dir=/mnt/data2/bhanu/datasets/all_data_for_ac_vc/BDL/wav
# ref_wav_path=/mnt/data2/bhanu/code-bases/ppg-vc/ref_wav/test/bhanu_voice.wav
ref_wav_path=/mnt/data2/bhanu/datasets/bac_vc_test_samples/NJS/wav/arctic_a0001.wav
# ref_wav_path=/mnt/data2/bhanu/code-bases/ppg-vc/ref_wav/test/arctic_a0292.wav
# src_wav_dir=/mnt/data2/bhanu/datasets/all_data_for_ac_vc/TXHC/wav
# ref_wav_path=/mnt/data2/bhanu/datasets/all_data_for_ac_vc/CLB/wav/arctic_a0001.wav
echo ${config}

# =============== One-shot VC ================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  exp_name="$(basename "${config}" .yaml)"
  echo Experiment name: "${exp_name}"
#   src_wav_dir="/home/shaunxliu/data/cmu_arctic/cmu_us_rms_arctic/wav"
#   ref_wav_path="/home/shaunxliu/data/cmu_arctic/cmu_us_slt_arctic/wav/arctic_a0001.wav"
  output_dir="vc_gen_wavs/$(basename "${config}" .yaml)"

  python convert_from_wav_transformer_single.py \
    --ppg2mel_model_train_config ${config} \
    --ppg2mel_model_file ${model_file} \
    --src_wav_dir "${src_wav_dir}" \
    --ref_wav_path "${ref_wav_path}" \
    -o "${output_dir}"
fi
