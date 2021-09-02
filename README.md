# Internal representations of neural networks
Analysis of internal representations of deep neural networks

### Installation

Clone the repository and install requirements as specified in `requirements.txt`.

Install PyTorch with GPU support

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

### Usage
This project encompasses multiple subprojects. All of them attempt to analyze
activations inside neural networks, but they differ in what are the ways to
approach such task. The subprojects are independent of each other. They
also differ in the way the results of the analysis are presented.

##### Simple example
Simple example of training a network and showing heatmaps. Complete
implementation, together with an analysis of the results can be found
in the `notebooks/simple.ipynb` notebook.

The network sometimes converges to a suboptimal solution, rerunning the training
procedure (as well as initialization of network) might be needed to reach
optimum.

##### Simple activations example
An example of neural network activations visualization for a simple,
handcrafted dataset. Complete implementation, together with an
analysis of the results can be found in the
`notebooks/simple_activations.ipynb` notebook.

##### Activations feedforward example
A more advanced example concerned with analysis of activations inside trained
neural networks. There are three ways to access this example:
- an analysis document of the results
- a jupyter notebook
- a script

This example is accompanied by a complete analysis in the format of a pdf.
This pdf (and corresponding latex document used to generate it) can be found
in the `docs/activations_feedforward` directory. The same way the reader is
encouraged to go through the jupyter notebook before running the script the
reader is encouraged to read this pdf before going through the jupyter
notebook.

The jupyter notebook also contains description of what is done and why, as well
as analysis of results. I would highly recommend to go over the notebook first
before using the script.

The `scripts/activations_feedforward.py` script can be run to acquire the same
type of results that can be seen in the notebook. However, there is no
accompanying description of what is done and no analysis. The results are
saved in the `out/activations_feedforward` directory in format suitable for
use in LaTeX. It is important to note that the script must be executed while
in the `scripts/` directory (i.e. the user first has to navigate to that
directory and after that execute the script); this is because the path to
the `out` directory is computed relative to the current working directory.

##### Activations convolutional example
The most important example. Activations of a convolutional neural network
are analyzed. The description and results can be found in the
`docs/neural_network_activations.pdf` file; the script used to obtain the
data is `scripts/activations_cnn.py`.

It is important to note that the script must be executed while
in the `scripts/` directory (i.e. the user first has to navigate to that
directory and after that execute the script); this is because the path to
the `out` directory is computed relative to the current working directory.


