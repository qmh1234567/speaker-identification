# speaker-identification
- 训练好的模型放在checkpoint文件夹中，由于deep speaker模型的大小超过100M，故没有上传，只上传了SE-RESCNN模型。
- run.py中，是使用两种模型融合的方式进行测试的，进行Speaker identification实验的结果是score=0.9519813，可以修改run,py中的代码，只用SE-RESCNN模型进行测试，结果为：score=0.9321678。
- 数据集是AIshell数据集，只使用了497人进行训练。
测试数据集内容是：4名说话人，每人使用mic、ios和android三种设备，分别于不同距离录制若干句语音，部分说话人录音环境有明显环境噪声。每个说话人共15句注册语音（每种设备5句，均为近场录制），测试语音则包含长句和短句，共8580句。
- 测试命令：
python run.py --enroll_dataset [注册集所在目录] --test_dataset [测试集所在目录] test
