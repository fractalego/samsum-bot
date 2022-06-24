1) Save torch model to onnx format:
```bash
$ python -m src.save_onnx_model
```

2) Run Docker image with tensorrt

3) In the docker image type
```bash
$ trtexec --onnx=onnx/gptj_model.onnx --saveEngine=test.engine
```