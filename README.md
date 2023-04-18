# Multi-Task Spatial-Temporal Graph Auto-Encoder for Hand Motion Denoising

This work (Multi-STGAE) extends [our conference paper](http://hubertshum.com/pbl_ismar2021hand.htm) presented at ISMAR 2021, which has been submitted to a journal. We plan to release the full code once it is accepted for publication. Currently, we provide the test code.

## Framework and Experiments

We recommend readers to watch the supplementary video of Multi-STGAE to gain a better understanding of the framework and to view the qualitative results.

Resources: [Multi-STGAE video](https://bhpan.buaa.edu.cn:443/link/8E94EF7ECE16BA78C7FC0237AB208475) | [STGAE website](http://hubertshum.com/pbl_ismar2021hand.htm) 

![](./imgs/overview.png)

<center>
    Framework overview of our proposed method Multi-STGAE: we utilize the prediction task to propose a multi-task framework for hand motion denoising. Through this framework, the denoised result is capable of preserving the temporal dynamics and the time delay problem can be greatly alleviated. In this way, it is possible to provide users with a satisfying experience during the interaction.
</center>





## Datasets



## Environments

- `LaTeX` tool

```bash
sudo apt-get install texlive-full
```

- `FFmpeg`

```bash
sudo apt-get install ffmpeg
```

- `pydot & graphviz`
```bash
sudo pip3 install pydot
sudo pip3 install graphviz
```

## Citation

If you find our work useful, we kindly request that you cite our paper to acknowledge our efforts and contributions. Thank you!

```latex
@INPROCEEDINGS{stage2021,
  author={Zhou, Kanglei and Cheng, Zhiyuan and Shum, Hubert P. H. and Li, Frederick W. B. and Liang, Xiaohui},
  booktitle={2021 IEEE International Symposium on Mixed and Augmented Reality (ISMAR)}, 
  title={STGAE: Spatial-Temporal Graph Auto-Encoder for Hand Motion Denoising}, 
  year={2021},
  volume={},
  number={},
  pages={41-49},
  doi={10.1109/ISMAR52148.2021.00018}
}
```

## Contact

Feel free to contact me via `zhoukanglei[at]qq.com`.