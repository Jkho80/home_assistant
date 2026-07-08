
import argparse
import json
import logging
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional
import time
import numpy as np
import torch
from flask import Flask, Response, jsonify, request
from gevent.pywsgi import WSGIServer

from hbm_runtime import HB_HBMRuntime
from robo_orchard_lab.models.holobrain.pipeline import HoloBrainInferencePipeline
from robo_orchard_lab.models.holobrain.processor import MultiArmManipulationInput
from robo_orchard_lab.models.holobrain.utils import apply_scale_shift, recompute

LOGGER = logging.getLogger("grasp_anything")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
APP = Flask(__name__)

CONTROL_HZ = 30
TRAINING_HZ = 30
INTERPOLATION = CONTROL_HZ / TRAINING_HZ

HB_DTYPE_MAP = {
    "U8": np.uint8,
    "S8": np.int8,
    "F16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
    "U16": np.uint16,
    "S16": np.int16,
    "S32": np.int32,
    "U32": np.uint32,
    "S64": np.int64,
    "U64": np.uint64,
    "BOOL8": np.bool_,
    "UINT8": np.uint8,
    "INT8": np.int8,
    "FLOAT16": np.float16,
    "FLOAT32": np.float32,
    "FLOAT64": np.float64,
    "UINT16": np.uint16,
    "INT16": np.int16,
    "INT32": np.int32,
    "UINT32": np.uint32,
    "INT64": np.int64,
    "UINT64": np.uint64,
    "BOOL": np.bool_,
}


@dataclass
class ServerConfig:
    encoder_hbm: str
    decoder_hbm: str
    model_dir: str
    inference_prefix: str = "grasp_anything_ro_our"
    server_name: str = "holobrain"
    host: str = "0.0.0.0"
    port: int = 8190
    num_joints: int = 7
    valid_action_step: int = 64
    torch_device: str = "cpu"
    return_interpolated_actions: bool = False
    verbose_timing: bool = True
    encoder_priority: int = 5
    decoder_priority: int = 5
    encoder_bpu_cores: Optional[List[int]] = None
    decoder_bpu_cores: Optional[List[int]] = None
    fixed_text_dir: Optional[str] = None
    expected_instruction: Optional[str] = "Clear objects into a basket."

    # Plan A: decoder uses precomputed float text_feature instead of encoder.hbm output
    use_precomputed_decoder_text_feature: bool = True
    decoder_text_feature_path: Optional[str] = None
    decoder_text_token_mask_path: Optional[str] = None

    @staticmethod
    def from_json(path: str) -> "ServerConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("encoder_bpu_cores") is None:
            data["encoder_bpu_cores"] = [0]
        if data.get("decoder_bpu_cores") is None:
            data["decoder_bpu_cores"] = [0]
        return ServerConfig(**data)


class HBMModel:
    def __init__(self, hbm_path: str, *, priority: int = 5, bpu_cores: Optional[List[int]] = None):
        self.hbm_path = hbm_path
        self.runtime = HB_HBMRuntime(hbm_path)
        if not self.runtime.model_names:
            raise RuntimeError(f"No model loaded from {hbm_path}")
        self.model_name = self.runtime.model_names[0]
        self.input_names = list(self.runtime.input_names[self.model_name])
        self.input_shapes = dict(self.runtime.input_shapes[self.model_name])
        self.input_dtypes = dict(self.runtime.input_dtypes[self.model_name])
        self.output_names = list(self.runtime.output_names[self.model_name])
        self.output_shapes = dict(self.runtime.output_shapes[self.model_name])
        self.output_dtypes = dict(self.runtime.output_dtypes[self.model_name])

        if bpu_cores is None:
            bpu_cores = [0]
        self.runtime.set_scheduling_params(
            priority={self.model_name: priority},
            bpu_cores={self.model_name: bpu_cores},
        )
        LOGGER.info(
            "Loaded HBM %s model=%s inputs=%s outputs=%s",
            hbm_path,
            self.model_name,
            self.input_names,
            self.output_names,
        )
        LOGGER.info(
            "HBM dtype summary for %s | input_dtypes=%s | output_dtypes=%s",
            self.model_name,
            {k: v.name for k, v in self.input_dtypes.items()},
            {k: v.name for k, v in self.output_dtypes.items()},
        )

    def _cast_input(self, name: str, arr: np.ndarray) -> np.ndarray:
        hb_dtype = self.input_dtypes[name].name
        np_dtype = HB_DTYPE_MAP.get(hb_dtype)
        if np_dtype is None:
            LOGGER.warning(
                "Unknown HBM dtype enum for input %s: %s, keep original numpy dtype=%s",
                name, hb_dtype, np.asarray(arr).dtype
            )
            np_dtype = np.asarray(arr).dtype
        arr = np.asarray(arr)
        if arr.dtype != np_dtype:
            arr = arr.astype(np_dtype, copy=False)
        return np.ascontiguousarray(arr)

    def run(self, feeds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        model_inputs: Dict[str, np.ndarray] = {}
        for name in self.input_names:
            if name not in feeds:
                raise KeyError(f"Missing HBM input `{name}` for model {self.hbm_path}")
            model_inputs[name] = self._cast_input(name, feeds[name])
        outputs = self.runtime.run({self.model_name: model_inputs})
        model_outputs = outputs[self.model_name]
        return {name: np.asarray(model_outputs[name]) for name in self.output_names}


class FixedTextBundle:
    def __init__(self, fixed_text_dir: str):
        fixed_text_dir = os.path.abspath(fixed_text_dir)
        self.fixed_text_dir = fixed_text_dir

        def _maybe_load(name: str):
            path = os.path.join(fixed_text_dir, f"{name}.npy")
            if os.path.exists(path):
                return np.load(path)
            return None

        self.input_ids = _maybe_load("input_ids")
        self.attention_mask = _maybe_load("attention_mask")
        self.position_ids = _maybe_load("position_ids")
        self.token_type_ids = _maybe_load("token_type_ids")
        self.text_token_mask = _maybe_load("text_token_mask")

        instruction_path = os.path.join(fixed_text_dir, "instruction.txt")
        self.instruction = None
        if os.path.exists(instruction_path):
            self.instruction = open(instruction_path, "r", encoding="utf-8").read().strip()

        if self.text_token_mask is None:
            raise RuntimeError(f"Missing text_token_mask.npy in {fixed_text_dir}")

    def check_instruction(self, incoming_instruction: str, expected_instruction: Optional[str] = None):
        expect = self.instruction if self.instruction else expected_instruction
        if expect and incoming_instruction != expect:
            LOGGER.warning(
                "Incoming instruction differs from fixed-text model instruction. incoming=%r expected=%r",
                incoming_instruction,
                expect,
            )


class PrecomputedDecoderTextProvider:
    def __init__(self, text_feature_path: str, text_token_mask_path: str):
        self.text_feature_path = os.path.abspath(text_feature_path)
        self.text_token_mask_path = os.path.abspath(text_token_mask_path)
        self.text_feature = np.load(self.text_feature_path)
        self.text_token_mask = np.load(self.text_token_mask_path)

        LOGGER.info(
            "Loaded precomputed decoder text bundle | text_feature=%s shape=%s dtype=%s | text_token_mask=%s shape=%s dtype=%s",
            self.text_feature_path, self.text_feature.shape, self.text_feature.dtype,
            self.text_token_mask_path, self.text_token_mask.shape, self.text_token_mask.dtype,
        )

    def get(self):
        return self.text_feature, self.text_token_mask


def _to_torch_device(x, device: torch.device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_torch_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_torch_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_torch_device(v, device) for v in x)
    return x


def unsqueeze_batch(data: dict):
    for k, v in list(data.items()):
        if isinstance(v, (torch.Tensor, np.ndarray)):
            data[k] = v[None]
        else:
            data[k] = [v]
    return data


def get_pixels_from_image_wh(image_wh, strides, device, dtype):
    image_w, image_h = int(image_wh[0]), int(image_wh[1])
    pixels = []
    for stride in strides:
        feature_wh = [image_w // stride, image_h // stride]
        u = torch.linspace(0, image_w - stride, feature_wh[0], device=device, dtype=dtype)
        v = torch.linspace(0, image_h - stride, feature_wh[1], device=device, dtype=dtype)
        u = u[None].tile(feature_wh[1], 1)
        v = v[:, None].tile(1, feature_wh[0])
        uv = torch.stack([u, v], dim=-1).flatten(0, 1)
        pixels.append(uv)
    return torch.cat(pixels, dim=0)[:, None]


class DeployDataPreprocessor:
    def __init__(
        self,
        model,
        pred_steps: int,
        hbm_image_wh,
        encoder_input_names: List[str],
        fixed_text_bundle: Optional[FixedTextBundle] = None,
        do_unsqueeze_batch: bool = False,
    ):
        self.model = model
        self.preprocessor = model.data_preprocessor.eval()
        try:
            self.preprocessor.to("cpu")
        except Exception:
            pass
        for _attr in ("device", "_device"):
            if hasattr(self.preprocessor, _attr):
                try:
                    setattr(self.preprocessor, _attr, torch.device("cpu"))
                except Exception:
                    pass

        self.decoder = model.decoder
        self.encoder_input_names = set(encoder_input_names)
        self.fixed_text_bundle = fixed_text_bundle
        self.strides = tuple(model.cfg.data_preprocessor["batch_transforms"][0]["stride"])
        self._pixels_cache = None
        self.pred_steps = pred_steps
        self.do_unsqueeze_batch = do_unsqueeze_batch
        self.hbm_image_wh = (int(hbm_image_wh[0]), int(hbm_image_wh[1]))

        LOGGER.info(
            "DeployDataPreprocessor initialized: image_wh=%s, strides=%s, encoder_inputs=%s, fixed_text=%s",
            self.hbm_image_wh,
            self.strides,
            sorted(self.encoder_input_names),
            self.fixed_text_bundle is not None,
        )

    def _expected_pixels_count(self):
        w, h = self.hbm_image_wh
        total = 0
        for stride in self.strides:
            total += (w // stride) * (h // stride)
        return total

    def _inject_fixed_text(self, data, device):
        if self.fixed_text_bundle is None:
            return
        bundle = self.fixed_text_bundle
        if bundle.text_token_mask is not None:
            data["text_token_mask"] = torch.from_numpy(bundle.text_token_mask).to(device)
        if "input_ids" in self.encoder_input_names and bundle.input_ids is not None:
            data["input_ids"] = torch.from_numpy(bundle.input_ids).to(device)
        if "attention_mask" in self.encoder_input_names and bundle.attention_mask is not None:
            data["attention_mask"] = torch.from_numpy(bundle.attention_mask).to(device)
        if "position_ids" in self.encoder_input_names and bundle.position_ids is not None:
            data["position_ids"] = torch.from_numpy(bundle.position_ids).to(device)
        if "token_type_ids" in self.encoder_input_names and bundle.token_type_ids is not None:
            data["token_type_ids"] = torch.from_numpy(bundle.token_type_ids).to(device)

    def __call__(self, data: dict):
        data = dict(data)
        data["projection_mat_inv"] = torch.linalg.inv(data["projection_mat"])
        if self.do_unsqueeze_batch:
            data = unsqueeze_batch(data)
        data = self.preprocessor(data, device = "cpu")

        if self._pixels_cache is None:
            self._pixels_cache = get_pixels_from_image_wh(
                self.hbm_image_wh,
                self.strides,
                device=data["imgs"].device,
                dtype=data["imgs"].dtype,
            )
        data["pixels"] = self._pixels_cache.clone()

        expected_pixels = self._expected_pixels_count()
        if data["pixels"].shape[0] != expected_pixels:
            raise RuntimeError(
                f"pixels shape mismatch after preprocessing: got {data['pixels'].shape[0]}, expected {expected_pixels}"
            )

        self._inject_fixed_text(data, data["imgs"].device)
        return data


def decode_request(request_data) -> MultiArmManipulationInput:
    images = {
        "left": [np.load(BytesIO(request_data["left_color"].read())).astype(np.uint8)],
        "right": [np.load(BytesIO(request_data["right_color"].read())).astype(np.uint8)],
        "middle": [np.load(BytesIO(request_data["middle_color"].read())).astype(np.uint8)],
    }
    depths = {
        "left": [np.load(BytesIO(request_data["left_depth"].read())).astype(np.float64) / 1000.0],
        "right": [np.load(BytesIO(request_data["right_depth"].read())).astype(np.float64) / 1000.0],
        "middle": [np.load(BytesIO(request_data["middle_depth"].read())).astype(np.float64) / 1000.0],
    }

    left_arm_state = np.load(BytesIO(request_data["left_arm_state"].read())).astype(np.float32)
    right_arm_state = np.load(BytesIO(request_data["right_arm_state"].read())).astype(np.float32)
    joint_state = np.concatenate([left_arm_state, right_arm_state], axis=-1)[None, :]

    intrinsics = np.eye(4, dtype=np.float64)[None].repeat(3, axis=0)
    intrinsics[0, :3] = np.load(BytesIO(request_data["left_intrinsic"].read())).astype(np.float64)
    intrinsics[1, :3] = np.load(BytesIO(request_data["right_intrinsic"].read())).astype(np.float64)
    intrinsics[2, :3] = np.load(BytesIO(request_data["middle_intrinsic"].read())).astype(np.float64)
    intrinsics = {"left": intrinsics[0], "right": intrinsics[1], "middle": intrinsics[2]}

    remaining_actions = (
        np.load(BytesIO(request_data["remaining_actions"].read())).astype(np.float32)[None]
        if request_data.get("remaining_actions") is not None
        else None
    )
    if remaining_actions is not None and remaining_actions.size > 0:
        remaining_actions = (
            torch.nn.functional.interpolate(
                torch.from_numpy(remaining_actions).permute(0, 2, 1),
                scale_factor=1 / INTERPOLATION,
                mode="linear",
                align_corners=True,
            )
            .permute(0, 2, 1)
            .numpy()
        )
    else:
        remaining_actions = None

    instruction = request_data.get("instruction", "Clear objects into a basket.")
    if hasattr(instruction, "read"):
        instruction = instruction.read().decode("utf-8")

    return MultiArmManipulationInput(
        image=images,
        depth=depths,
        history_joint_state=joint_state,
        intrinsic=intrinsics,
        instruction=instruction,
        remaining_actions=remaining_actions,
        delay_horizon=int(request_data.get("delay_horizon", 0)),
    )

t7 = None

class HBMHoloBrainRunner:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.torch_device = torch.device("cuda" if (cfg.torch_device == "auto" and torch.cuda.is_available()) else cfg.torch_device)

        self.pipeline = HoloBrainInferencePipeline.load_pipeline(
            directory=cfg.model_dir,
            device=str(self.torch_device),
            load_weights=False,
            load_impl="native",
            inference_prefix=cfg.inference_prefix,
        )
        self.t7 = None
        self.pipeline.model.eval()
        try:
            self.pipeline.model.to("cpu")
        except Exception:
            pass
        if hasattr(self.pipeline.model, "data_preprocessor"):
            try:
                self.pipeline.model.data_preprocessor.to("cpu")
            except Exception:
                pass
            for _attr in ("device", "_device"):
                if hasattr(self.pipeline.model.data_preprocessor, _attr):
                    try:
                        setattr(self.pipeline.model.data_preprocessor, _attr, torch.device("cpu"))
                    except Exception:
                        pass

        self.encoder_model = HBMModel(cfg.encoder_hbm, priority=cfg.encoder_priority, bpu_cores=cfg.encoder_bpu_cores)
        self.decoder_model = HBMModel(cfg.decoder_hbm, priority=cfg.decoder_priority, bpu_cores=cfg.decoder_bpu_cores)

        fixed_text_dir = cfg.fixed_text_dir
        if not fixed_text_dir:
            auto_dir = os.path.join(os.path.dirname(os.path.abspath(cfg.encoder_hbm)), "fixed_text")
            if os.path.isdir(auto_dir):
                fixed_text_dir = auto_dir

        self.fixed_text_bundle = None
        if fixed_text_dir and os.path.isdir(fixed_text_dir):
            self.fixed_text_bundle = FixedTextBundle(fixed_text_dir)
            LOGGER.info("Loaded fixed_text bundle from %s", fixed_text_dir)

        self.precomputed_text_provider = None
        if cfg.use_precomputed_decoder_text_feature:
            if not cfg.decoder_text_feature_path or not cfg.decoder_text_token_mask_path:
                raise RuntimeError(
                    "Plan A enabled but decoder_text_feature_path / decoder_text_token_mask_path not set"
                )
            self.precomputed_text_provider = PrecomputedDecoderTextProvider(
                cfg.decoder_text_feature_path,
                cfg.decoder_text_token_mask_path,
            )

        imgs_shape = self.encoder_model.input_shapes["imgs"]
        hbm_image_wh = (int(imgs_shape[-1]), int(imgs_shape[-2]))

        self.deploy_preprocessor = DeployDataPreprocessor(
            self.pipeline.model,
            pred_steps=self.pipeline.model.decoder.pred_steps,
            hbm_image_wh=hbm_image_wh,
            encoder_input_names=self.encoder_model.input_names,
            fixed_text_bundle=self.fixed_text_bundle,
            do_unsqueeze_batch=False,
        )

        LOGGER.info("Using torch_device=%s", self.torch_device)
        LOGGER.info("Encoder HBM inputs=%s outputs=%s", self.encoder_model.input_names, self.encoder_model.output_names)
        LOGGER.info("Decoder HBM inputs=%s outputs=%s", self.decoder_model.input_names, self.decoder_model.output_names)
        LOGGER.info("Plan A enabled=%s", self.precomputed_text_provider is not None)

    @torch.inference_mode()
    def infer(self, request_data):
        timer = {}
        
        if self.cfg.verbose_timing:
            t0 = time.perf_counter()

        if self.t7 != None:
            print("Last time:",t0 - self.t7)

        model_input = decode_request(request_data)
        if self.fixed_text_bundle is not None:
            self.fixed_text_bundle.check_instruction(model_input.instruction, self.cfg.expected_instruction)

        if self.cfg.verbose_timing:
            t1 = time.perf_counter()

        processed = self.pipeline.processor.pre_process(model_input)
        data = self.deploy_preprocessor(processed)
        data = _to_torch_device(data, self.torch_device)

        if self.cfg.verbose_timing:
            t2 = time.perf_counter()

        encoder_feeds = {name: data[name].detach().cpu().numpy() for name in self.encoder_model.input_names}
        encoder_outs = self.encoder_model.run(encoder_feeds)
        image_feature_np = np.asarray(encoder_outs["image_feature"], dtype=np.float32)
        robot_feature_np = np.asarray(encoder_outs["robot_feature"], dtype=np.float32)

        # Plan A: do NOT use encoder.hbm text_feature for decoder
        if self.precomputed_text_provider is not None:
            text_feature_np, text_token_mask_np = self.precomputed_text_provider.get()
            text_feature_np = np.asarray(text_feature_np, dtype=np.float32)
            text_token_mask_np = np.asarray(text_token_mask_np, dtype=np.uint8)
        else:
            text_feature_np = np.asarray(encoder_outs["text_feature"], dtype=np.float32)
            # self.get_logger().info("Using encoder OUTPUT as decoder INPUT")
            text_token_mask_source = (
                self.fixed_text_bundle.text_token_mask
                if self.fixed_text_bundle is not None
                else data["text_token_mask"].detach().cpu().numpy()
            )
            text_token_mask_np = np.asarray(text_token_mask_source, dtype=np.uint8)

        if self.cfg.verbose_timing:
            t3 = time.perf_counter()

        decoder = self.pipeline.model.decoder
        # self.get_logger().info(f"text feat shape {text_feature_np.shape}")

        hist_robot_state = apply_scale_shift(data["hist_robot_state"], data.get("joint_scale_shift"))
        bs, _, num_joint, state_dims = hist_robot_state.shape
        noise_type = decoder.noise_type

        noisy_action = decoder.sample_noise(
            [bs, decoder.pred_steps, num_joint, state_dims],
            hist_robot_state,
            noise_type,
        )

        test_scheduler = decoder.test_noise_scheduler
        test_scheduler.set_timesteps(decoder.num_inference_timesteps, device=hist_robot_state.device)

        remaining_actions = None
        delay_horizon = None
        if (
            decoder.async_inference_plugin is not None
            and "remaining_actions" in data
            and "delay_horizon" in data
        ):
            remaining_actions = data["remaining_actions"][0].to(hist_robot_state).unsqueeze(-1)
            remaining_actions = apply_scale_shift(remaining_actions, data["joint_scale_shift"])
            delay_horizon = data["delay_horizon"][0]

        joint_relative_pos_np = data["joint_relative_pos"].detach().cpu().numpy().astype(np.int32, copy=False)
        joint_mask_np = data["joint_mask"].detach().cpu().numpy().astype(np.uint8, copy=False)

        static_decoder_feeds = {
            "image_feature": image_feature_np,
            "robot_feature": robot_feature_np,
            "text_feature": text_feature_np,
            "text_token_mask": text_token_mask_np,
            "joint_relative_pos": joint_relative_pos_np,
            "hist_robot_state": data["hist_robot_state"].detach().cpu().numpy(),
            "joint_scale_shift": data["joint_scale_shift"].detach().cpu().numpy(),
            "joint_mask": joint_mask_np,
        }

        if self.cfg.verbose_timing:
            t4 = time.perf_counter()
        decoder_hbm_times = []  # 新增：记录每次HBM推理时间

        for t in test_scheduler.timesteps:
            decoder_input = recompute(noisy_action, data) if not noise_type.endswith("pose") else noisy_action
            decoder_feeds = dict(static_decoder_feeds)
            decoder_feeds["noisy_action"] = decoder_input.detach().cpu().numpy()
            decoder_feeds["timestep"] = np.asarray([int(t)], dtype=np.int32)
            if self.cfg.verbose_timing:
                hbm_start = time.perf_counter()
            

            pred_np = self.decoder_model.run(decoder_feeds)["pred_action"]

            if self.cfg.verbose_timing:
                hbm_end = time.perf_counter()
                decoder_hbm_times.append((hbm_end - hbm_start))
                
            pred = torch.from_numpy(np.asarray(pred_np, dtype=np.float32)).to(self.torch_device)

            if remaining_actions is not None:
                pred = decoder.async_inference_plugin(pred, remaining_actions, delay_horizon)

            if not noise_type.endswith("pose"):
                next_joint = test_scheduler.step(pred[..., :1], t, noisy_action[..., :1]).prev_sample
                noisy_action = torch.cat([next_joint, pred[..., 1:]], dim=-1)
            else:
                noisy_action = test_scheduler.step(pred, t, noisy_action).prev_sample

        if self.cfg.verbose_timing:
            t5 = time.perf_counter()

        pred_actions = noisy_action.unsqueeze(1)
        model_outs = {"pred_actions": pred_actions}
        decoder_pp = decoder.post_process(model_outs, data)
        output = self.pipeline.processor.post_process(decoder_pp, data)
        actions = output.action

        if self.cfg.return_interpolated_actions:
            actions = torch.nn.functional.interpolate(
                actions.permute(1, 0)[None],
                scale_factor=INTERPOLATION,
                mode="linear",
                align_corners=True,
            )[0].permute(1, 0)
            actions = actions[: int(self.cfg.valid_action_step * INTERPOLATION)]
        else:
            actions = actions[: int(self.cfg.valid_action_step)]

        if self.cfg.verbose_timing:
            t6 = time.perf_counter()
            timer["decode_request"] = t1 - t0
            timer["preprocess"] = t2 - t1
            timer["encoder_hbm"] = t3 - t2
            timer["rollout_setup"] = t4 - t3
            timer["decoder_rollout_total"] = t5 - t4
            timer["postprocess"] = t6 - t5
            timer["total"] = t6 - t0
            if decoder_hbm_times:
                timer["decoder_hbm_avg"] = round(sum(decoder_hbm_times) / len(decoder_hbm_times), 2)
                timer["decoder_hbm_min"] = round(min(decoder_hbm_times), 2)
                timer["decoder_hbm_max"] = round(max(decoder_hbm_times), 2)
                timer["decoder_hbm_total"] = round(sum(decoder_hbm_times), 2)
                timer["num_inference_steps"] = len(decoder_hbm_times)
            LOGGER.info("timing=%s", {k: round(v * 1000, 2) for k, v in timer.items()})
        self.t7  = time.perf_counter()
        while self.t7 - t0 < 0.75:
            self.t7 = time.perf_counter()        
        return {
            "left_arm_actions": actions[:, : self.cfg.num_joints].cpu().numpy().tolist(),
            "right_arm_actions": actions[:, self.cfg.num_joints :].cpu().numpy().tolist(),
            "action_horizon": len(actions),
        }


def build_app(runner: HBMHoloBrainRunner, cfg: ServerConfig) -> Flask:
    app = APP

    @app.route(f"/{cfg.server_name}", methods=["POST"])
    def model_infer():
        try:
            data = {**request.files, **request.form}
            required_keys = [
                "left_color", "middle_color", "right_color",
                "left_depth", "middle_depth", "right_depth",
                "left_intrinsic", "middle_intrinsic", "right_intrinsic",
                "left_arm_state", "right_arm_state", "instruction",
            ]
            for key in required_keys:
                if key not in data:
                    return jsonify({"error": f"Missing key: {key}"}), 400

            res = runner.infer(data)
            return Response(json.dumps(res), mimetype="application/json")
        except Exception as e:
            logging.exception("Error in endpoint: %s", e)
            return jsonify({"error": str(e)}), 500

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_config", type=str, default="server_grasp_anything.json")
    args = parser.parse_args()

    cfg = ServerConfig.from_json(args.server_config)
    runner = HBMHoloBrainRunner(cfg)
    app = build_app(runner, cfg)

    LOGGER.info("HBM model server %s started on %s:%s", cfg.server_name, cfg.host, cfg.port)
    http_server = WSGIServer((cfg.host, cfg.port), app)
    http_server.serve_forever()


if __name__ == "__main__":
    main()
