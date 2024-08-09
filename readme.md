# 项目说明
Various artificial intelligence (AI) algorithms have been developed for autonomous vehicles (AVs) to support environmental perception, decision making and automated driving in real-world scenarios. Existing AI methods, such as deep learning and deep reinforcement learning, have been criticized due to their black box nature. Explainable AI technologies are important for assisting users in understanding vehicle behaviors to ensure that users trust, accept, and rely on AI devices. In this paper, an explainable $Q$-learning method for AV longitudinal control is proposed. First, AI control of AVs is realized by constructing a deep $Q$-network (DQN) with an intelligent driver model, with the control objective maximizing vehicle speed while preventing collisions. Then, a deep explainer for humans is developed via a Shapley additive explanation (SHAP), and a novel positive SHAP method that defines new base values is proposed to explain how individual state features contribute to decisions. Finally, statistical analyses and intuitive explanations are quantified based on SHAP tools to improve clarity. Elaborate numerical simulations are conducted to demonstrate the effectiveness of the proposed algorithm.

![EIDG](/home/ubuntu/limeng/projecto_file/Pos_Shap/framework.png)

# 项目使用
## 前提条件
ubuntu 18.04 (windows 系统也能够支持)

安装conda

## 库安装

* 使用conda创建python=3.8虚拟环境  

* conda create -n your_env_name python=3.8

* conda activate your_env_name

* pip install -r requirements.txt

将该库下载到本地

## 示例
*  局部解释和全局解释 draw_figure.ipynb
*  shapley值量化比较 shapley_compare.ipynb