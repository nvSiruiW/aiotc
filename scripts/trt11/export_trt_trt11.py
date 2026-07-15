#!/usr/bin/env python3
"""ONNX -> TensorRT engine for TensorRT 11.x (strongly-typed, no builder flags).

Adaptation of scripts/export_trt.py for the TensorRT shipped in this JetPack/CUDA
stack (TRT 11.1). In TRT 11.x the classic weakly-typed precision path is GONE:
  * trt.IInt8EntropyCalibrator2 / any IInt8Calibrator  -> REMOVED
  * BuilderFlag.FP16 / BuilderFlag.INT8 / BuilderFlag.BF16 -> REMOVED
  * trtexec --fp16/--int8/--calib -> REMOVED
Networks are strongly-typed by default; precision must live in the ONNX graph:
  * FP16  -> a float16 ONNX (onnxconverter_common.float16)
  * INT8  -> a Q/DQ ONNX with real calibrated scales (onnxruntime static quant)
So calibration now happens in ONNX Runtime over the SAME real IMU windows the old
entropy calibrator would have used; TRT then honours the embedded Q/DQ scales.

Usage:
  python export_trt_trt11.py --onnx onnx/ronin_resnet18.onnx \
      --out engines/orin/ronin_resnet18 --precisions fp16,int8 \
      --input-name imu --input-shape 1,6,200 --calib calib/imu_calib_200.npy
"""
import argparse, os, numpy as np, tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)            # strongly-typed by default in TRT 11
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("  ONNX parse error:", parser.get_error(i))
            raise RuntimeError("ONNX parse failed")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    ser = builder.build_serialized_network(network, config)
    if ser is None:
        raise RuntimeError("engine build returned None")
    return bytes(ser)                              # IHostMemory -> bytes


def make_fp16_onnx(src, dst):
    import onnx
    from onnxconverter_common import float16
    m = onnx.load(src)
    m16 = float16.convert_float_to_float16(m, keep_io_types=False)
    onnx.save(m16, dst)
    return dst


class _CalibReader:
    """onnxruntime CalibrationDataReader over an [N,C,L] npy of real windows."""
    def __init__(self, arr, input_name):
        from onnxruntime.quantization import CalibrationDataReader  # noqa
        self.input_name = input_name
        self._it = iter([{input_name: arr[i:i+1].astype(np.float32)} for i in range(len(arr))])
    def get_next(self):
        return next(self._it, None)


def make_int8_qdq_onnx(src, dst, calib_arr, input_name):
    from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationMethod
    # symmetric int8 activations+weights, per-channel -> TRT-friendly QDQ
    quantize_static(
        model_input=src, model_output=dst,
        calibration_data_reader=_CalibReader(calib_arr, input_name),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        per_channel=True, calibrate_method=CalibrationMethod.MinMax,
        # TRT requires symmetric int8 (zero_point==0); force it or TRT's ONNX
        # parser rejects the QuantizeLinear nodes ("Non-zero zero point ...").
        # TRT rejects Int32 QDQ on conv bias; leave bias unquantized (TRT folds it).
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True,
                       "QuantizeBias": False},
    )
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True, help="output prefix, e.g. engines/orin/ronin_resnet18")
    ap.add_argument("--precisions", default="fp16,int8")
    ap.add_argument("--input-name", default="imu")
    ap.add_argument("--input-shape", default="1,6,200")
    ap.add_argument("--calib", default=None, help="npy [N,C,L] of real calibration windows (int8)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    calib_arr = None
    if args.calib and os.path.exists(args.calib):
        calib_arr = np.load(args.calib).astype(np.float32)

    for prec in [p.strip() for p in args.precisions.split(",")]:
        try:
            if prec == "fp32":
                onnx_for_trt = args.onnx
            elif prec == "fp16":
                onnx_for_trt = make_fp16_onnx(args.onnx, f"{args.out}_fp16.onnx")
            elif prec == "int8":
                if calib_arr is None:
                    raise RuntimeError("int8 requires --calib with real windows (accuracy must be measured)")
                onnx_for_trt = make_int8_qdq_onnx(args.onnx, f"{args.out}_int8.onnx", calib_arr, args.input_name)
            else:
                raise RuntimeError(f"unknown precision {prec}")
            ser = build_engine(onnx_for_trt)
            path = f"{args.out}_{prec}.plan"
            with open(path, "wb") as f:
                f.write(ser)
            print(f"  [{prec}] OK -> {path}  ({len(ser)//1024} KB)")
        except Exception as e:
            print(f"  [{prec}] FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
