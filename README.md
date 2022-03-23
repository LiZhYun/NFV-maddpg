# NFV Multi-Agent Deep Deterministic Policy Gradient (NFVMADDPG)

为后续发文所写的实验代码，MADDPG参考了以下论文及仓库:
[Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf).
[Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs).


**贡献:** 根据NFV环境需要，根据Gym自行构建了适合NFV的多智能体环境。 

## 安装

- 首先`cd` 到项目根目录， 并在cmd中输入`pip install -e .`(安装项目中的包)

- 实验环境: Python (3.6), OpenAI gym (0.10.5), tensorflow (1.8.0), numpy (1.14.5)

## 安装NFV环境: Multi-Agent NFV Environments

相关环境地址为：
[Multi-Agent NFV Environments (MNFVE)](https://github.com/LiZhYun/multiagent-nfv-envs).

- 根据`README`下载并安装。 [点击此处](https://github.com/LiZhYun/multiagent-nfv-envs)

- 这样可以保证multiagent-nfv-envs在您的搜索路径中

## 运行

- 在项目根目录执行 `python  experiments/train.py`


## Paper citation

<pre>
@article{lowe2017multi,
  title={Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments},
  author={Lowe, Ryan and Wu, Yi and Tamar, Aviv and Harb, Jean and Abbeel, Pieter and Mordatch, Igor},
  journal={Neural Information Processing Systems (NIPS)},
  year={2017}
}
</pre>
