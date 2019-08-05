# test
python newpreprocess.py --in_dir /data/verify/Verify/wav --pk_dir /data/verify/Verify/pickle --data_type aishell
# enroll
python newpreprocess.py --in_dir /data/verify/Train/wav/ --pk_dir /data/verify/Train/pickle --data_type aishell


python run.py --enroll_dataset /data/verify/Train --test_dataset /data/verify/Verify --mode_type test

python run.py --enroll_dataset /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/AIShell/SI_devdataset/pickle --test_dataset /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/AIShell/SI_devdataset/pickle --mode_type test