#!/usr/bin/env python3
"""ONNX -> TensorRT engine (fp32 / fp16 / int8+PTQ calibration).

Usage:
  python export_trt.py --onnx onnx/ronin_resnet18.onnx \
      --out engines/<device>/ronin_resnet18 \
      --precisions fp32,fp16,int8 \
      --input-name imu --calib calib/ronin_resnet18_calib.npy

Notes:
- int8 latency/energy is valid regardless of calibration quality; only ACCURACY
  needs a real calibration set. If --calib is omitted, random calibration data
  is synthesized (fine for a latency/energy benchmark; do NOT use for accuracy).
- Uses torch CUDA tensors as calibrator buffers (no pycuda dependency).
"""
import argparse, os, numpy as np, tensorrt as trt, torch

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class NpyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, arr, input_name, cache_path):
        super().__init__()
        self.batches = [np.ascontiguousarray(arr[i:i+1]) for i in range(len(arr))]
        self.input_name, self.cache_path, self.i, self._hold = input_name, cache_path, 0, None

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.i >= len(self.batches):
            return None
        self._hold = torch.from_numpy(self.batches[self.i].astype(np.float32)).cuda()
        self.i += 1
        return [int(self._hold.data_ptr())]

    def read_calibration_cache(self):
        return open(self.cache_path, "rb").read() if os.path.exists(self.cache_path) else None

    def write_calibration_cache(self, cache):
        open(self.cache_path, "wb").write(cache)


def build(onnx_path, precision, input_name, calib_arr, cache_path, workspace_gb=4):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)  # TRT10/11: explicit batch is default
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("  ONNX parse error:", parser.get_error(i))
            raise RuntimeError("ONNX parse failed")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = NpyCalibrator(calib_arr, input_name, cache_path)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"engine build failed for {precision}")
    return serialized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True, help="output prefix, e.g. engines/blackwell/ronin_resnet18")
    ap.add_argument("--precisions", default="fp32,fp16,int8")
    ap.add_argument("--input-name", default="imu")
    ap.add_argument("--input-shape", default="1,6,200", help="for random calib if --calib missing")
    ap.add_argument("--calib", default=None, help="npy of calibration inputs [N,C,L]")
    ap.add_argument("--n-calib", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    shape = tuple(int(s) for s in args.input_shape.split(","))
    if args.calib and os.path.exists(args.calib):
        calib_arr = np.load(args.calib).astype(np.float32)
        print(f"calib: loaded {len(calib_arr)} real windows from {args.calib}")
    else:
        calib_arr = np.random.randn(args.n_calib, *shape[1:]).astype(np.float32)
        print(f"calib: SYNTHETIC {args.n_calib} windows (latency/energy valid; NOT for accuracy)")

    for prec in args.precisions.split(","):
        prec = prec.strip()
        cache = f"{args.out}_{prec}.calib"
        try:
            ser = build(args.onnx, prec, args.input_name, calib_arr, cache)
            path = f"{args.out}_{prec}.plan"
            open(path, "wb").write(ser)
            print(f"  [{prec}] OK -> {path}  ({os.path.getsize(path)//1024} KB)")
        except Exception as e:
            print(f"  [{prec}] FAILED: {e}")


if __name__ == "__main__":
    main()
