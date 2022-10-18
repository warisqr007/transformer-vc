. ./path.sh || exit 1;
export CUDA_VISIBLE_DEVICES=1

stage=1
stop_stage=1
config=/home/grads/b/bhanu/ppg-vc/conf/own.yaml
model_file=/home/grads/b/bhanu/ppg-vc/ckpt/own_seed0/step_673000.pth
src_wav_dir=/home/grads/b/bhanu/ppg-vc/source_wav/test
ref_wav_path=/home/grads/b/bhanu/ppg-vc/ref_wav/test/arctic_a0292.wav
echo ${config}

# =============== One-shot VC ================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  exp_name="$(basename "${config}" .yaml)"
  echo Experiment name: "${exp_name}"
#   src_wav_dir="/home/shaunxliu/data/cmu_arctic/cmu_us_rms_arctic/wav"
#   ref_wav_path="/home/shaunxliu/data/cmu_arctic/cmu_us_slt_arctic/wav/arctic_a0001.wav"
  output_dir="vc_gen_wavs/$(basename "${config}" .yaml)"

  python convert_from_wav.py \
    --ppg2mel_model_train_config ${config} \
    --ppg2mel_model_file ${model_file} \
    --src_wav_dir "${src_wav_dir}" \
    --ref_wav_path "${ref_wav_path}" \
    -o "${output_dir}"
fi
