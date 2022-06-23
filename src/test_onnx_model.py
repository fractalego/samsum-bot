import onnx
import os


_path = os.path.dirname(__file__)
_save_path_onnx = os.path.join(_path, "../samsumbot_onnx/gptj_model.onnx")

if __name__ == "__main__":
    onnx_model = onnx.load(_save_path_onnx)
    onnx.checker.check_model(onnx_model)
