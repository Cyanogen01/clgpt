# clgpt
### Code release for CL-GPT (Convolutional Long-term structure with Gaussian Process Transfer)


## Resources
### [__Paper__](https://ieeexplore.ieee.org/abstract/document/11051099) | [__Wechat Article__](https://mp.weixin.qq.com/s/yIpLTAMrvjQIBs-GrgO10g) | [__DIB-Lab Webcite__](https://dib-lab.com/)


## Short Summary
An end-to-end scalable deep recurrent structure with fast transfer learning is proposed to accurately estimate SOC for lithium-ion batteries at different temperatures. First, convolutional and recurrent neural networks are employed to catch nonlinear temporal dependency within measurements. Second, a GPR layer is concatenated after neural networks, so that the estimation can be quantified with uncertainty while retaining nonlinear expressive ability. Then, a non-parametric fast transfer learning is designed to realize fast transfer between different temperatures. Next, structured sparse approximations and a semi-stochastic gradient procedure are established for scalable training. Finally, the accuracy and fast transfer of the proposed structure are verified through comparison. The structure demonstrates the state-of-the-art performance on estimation accuracy and efficiency with transfer learning faster than fine-tuning strategy by two orders of magnitude.

## Poster
<img width="3024" height="4536" alt="poster" src="https://github.com/user-attachments/assets/5a874c33-6bf0-41ec-8a98-69e790347bb4" />
