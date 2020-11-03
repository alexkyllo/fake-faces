# fake-faces

Fake Face Detection Project for CSS 581 Machine Learning at UW Bothell.

## Installing

Install [Poetry](https://python-poetry.org/) and then type `poetry install`.

Install `make`: on Ubuntu, `sudo apt install make`

Install a LaTeX distribution, such as `TeXLive` or `MiKTeX`, that includes `pdflatex`
to compile the PDF report.

### GPU Support

For Tensorflow GPU Support, follow the instructions at
[GPU Support|Tensorflow](https://www.tensorflow.org/install/gpu)

Additionally, as TF complained that `libcublas.so.10` was missing,
I had to `sudo apt install libcublas10` and add the following to `~/.bashrc`:

``` shell
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-10.2/targets/x86_64-linux/lib:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
```

Alternatively, you can `conda install tensorflow-gpu` along with the other
dependencies listed in `pyproject.toml`.

## Testing

`pytest`
