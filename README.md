# Internal representations of neural networks
Analysis of internal representations of deep neural networks

### Usage
Clone the repository and install requirements as specified in `requirements.txt`.

Install PyTorh with GPU support

```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

or without

```
pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

Install this package

```
pip intall -e .
```


##### Simple example
Simple example of training a network and showing heatmaps can be accessed two ways:
- using jupyter notebook
- calling `python src/simple.py` and following instructions in console

The network sometimes converges to a suboptimal solution, rerunning the training procedure (as well as initialization of network) might be needed to reach optimum.


##### Simple activations example
An example of neural network activations visualization is currently only available
as Jupyter notebook. 
