Compatibility stand-ins used to run the official ms-pred GLACIER code (MIT, github.com/coleygroup/ms-pred) on
Kaggle's Python 3.13 image, where DGL does not install:

* `dgl_shim.py` — installed as the `dgl` package: only the import surface is provided. GLACIER's Graphormer encoder
  never builds DGL graphs (root_encode = "graphormer"); any actual DGL call raises with the API name.
* `torch_scatter_shim.py` — installed as `torch_scatter` when the real package is missing: scatter_{add,sum,mean,max,
  min} on top of torch.Tensor.scatter_reduce (GLACIER inference uses scatter_max values only).
