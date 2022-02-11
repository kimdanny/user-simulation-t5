# user-simulation
User Simulation by utterance generation, and action+satisfaction score prediction

T5 MTL

## EC2 Environment setting
**Deep Learning AMI (Amazon Linux 2) Version 57.0**  
**source activate pytorch_p38**
**source activate pytorch_p37**

Please use one of the following commands to start the required environment with the framework of your choice:  

    for TensorFlow 2.7 with Python3.8 (CUDA 11.2 and Intel MKL-DNN) ________________________________ source activate tensorflow2_p38
    for PyTorch 1.10 with Python3.8 (CUDA 11.1 and Intel MKL) __________________________________________ source activate pytorch_p38
    for AWS MX 1.8 (+Keras2) with Python3.7 (CUDA 11.0 and Intel MKL-DNN) ________________________________ source activate mxnet_p37

    for AWS MX(+AWS Neuron) with Python3 ______________________________________________________ source activate aws_neuron_mxnet_p36
    for TensorFlow(+AWS Neuron) with Python3 _____________________________________________ source activate aws_neuron_tensorflow_p36
    for PyTorch (+AWS Neuron) with Python3 __________________________________________________ source activate aws_neuron_pytorch_p36

    for TensorFlow 2(+Amazon Elastic Inference) with Python3 ______________________________ source activate amazonei_tensorflow2_p36
    for PyTorch 1.5.1 (+Amazon Elastic Inference) with Python3 _________________________ source activate amazonei_pytorch_latest_p37
    for AWS MX(+Amazon Elastic Inference) with Python3 __________________________________________ source activate amazonei_mxnet_p36
    for base Python3 (CUDA 11.0) ___________________________________________________________________________ source activate python3

To automatically activate base conda environment upon login, run: 'conda config --set auto_activate_base true'

Official Conda User Guide: https://docs.conda.io/projects/conda/en/latest/user-guide/  
AWS Deep Learning AMI Homepage: https://aws.amazon.com/machine-learning/amis/  
Developer Guide and Release Notes: https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html  
Support: https://forums.aws.amazon.com/forum.jspa?forumID=263  
For a fully managed experience, check out Amazon SageMaker at https://aws.amazon.com/sagemaker  
When using INF1 type instances, please update regularly using the instructions at: https://github.com/aws/aws-neuron-sdk/tree/master/release-notes  
Security scan reports for python packages are located at: /opt/aws/dlami/info/  
