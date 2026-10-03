from __future__ import annotations

import os
import re
import shlex
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import comfy.model_management
import folder_paths
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

PERF_RE = re.compile(r"\[\s*Prompt:[^|\]]+\|\s*Generation:[^\]]+\]")
START_THINKING = "[Start thinking]"
END_THINKING = "[End thinking]"
START_REDACTED = "[Start thinking (redacted)]"
END_REDACTED = "[End thinking (redacted)]"
RAW_EOF_MARKERS = ("\n> EOF by user", "\n\n> EOF by user")
LLAMA_SEED_MODULUS = 2 ** 32

MODEL_SCAN_KEYS = ("LLM", "llm", "clip", "text_encoders", "diffusion_models", "unet", "checkpoints")
MODEL_CACHE_TTL = 30.0
_model_cache = {"expires": 0.0, "models": []}


def list_gguf_models() -> list[str]:
    now = time.time()
    if now < _model_cache["expires"]:
        return _model_cache["models"]
    found = set()
    registry = folder_paths.folder_names_and_paths
    for key in MODEL_SCAN_KEYS:
        for root in registry.get(key, ([], set()))[0]:
            root_dir = Path(root)
            if not root_dir.is_dir():
                continue
            try:
                found.update(str(p) for p in root_dir.rglob("*.gguf"))
            except OSError:
                continue
    _model_cache["models"] = sorted(found, key=lambda p: (len(p), p))
    _model_cache["expires"] = now + MODEL_CACHE_TTL
    return _model_cache["models"]


def resolve_model_path(model: str, model_path_override: str) -> Path:
    raw = (model_path_override or "").strip() or (model or "").strip()
    if not raw:
        raise ValueError("No GGUF selected. Pick model in the combo or put an absolute path into model_path_override.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(folder_paths.models_dir) / path
    if not path.is_file():
        raise ValueError(f"GGUF model not found: {path}")
    return path


def _exe_name() -> str:
    return "llama-cli.exe" if os.name == "nt" else "llama-cli"


def _cli_from(candidate: str) -> str | None:
    if not candidate:
        return None
    path = Path(candidate).expanduser()
    if path.is_dir():
        path = path / _exe_name()
    return str(path) if path.is_file() else None


def _read_pack_config(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    except OSError:
        pass
    return ""


def find_llama_cli(explicit: str = "") -> str:
    pack_dir = Path(__file__).resolve().parent
    for candidate in (
        explicit,
        os.environ.get("LLAMA_CLI"),
        _read_pack_config(pack_dir / "llama_path.txt"),
        shutil.which("llama-cli"),
        shutil.which(_exe_name()),
    ):
        found = _cli_from(candidate or "")
        if found:
            return found
    # llama.cpp bin folders shipped by us or dropped by other packs (vendor first, shallow wins)
    for root in (pack_dir / "vendor", pack_dir, pack_dir.parent / "ComfyUI-LLM-text-processor"):
        if root.is_dir():
            for hit in sorted(root.rglob(_exe_name())):
                return str(hit)
    # loose llama-*-bin-win-* extracts on drive roots
    for drive in "CDEF":
        if not os.path.exists(f"{drive}:\\"):
            continue
        for hit in sorted(Path(f"{drive}:\\").glob(f"llama*/{_exe_name()}")):
            return str(hit)
    raise RuntimeError(
        "llama-cli not found. Point llama_cli_path to your llama.cpp bin directory (or llama-cli.exe), "
        "or write that path into custom_nodes/ComfyUI-LlamaCli-TextGenerate/llama_path.txt."
    )


def find_sibling_exe(cli_path: str, name: str) -> str:
    exe = Path(cli_path)
    suffix = ".exe" if os.name == "nt" else ""
    sibling = exe.parent / (name + suffix)
    if sibling.is_file():
        return str(sibling)
    found = shutil.which(name)
    if found:
        return found
    raise RuntimeError(f"{sibling.name} not found next to llama-cli; a full llama.cpp bin dir is required.")


def normalize_llama_seed(seed: int) -> int:
    seed = int(seed)
    if 0 <= seed < LLAMA_SEED_MODULUS:
        return seed
    return seed % LLAMA_SEED_MODULUS


def gguf_has_mtp_head(path: Path) -> bool:
    # MTP/extra-prediction heads show up as tensor names like mtp.* or predict.*
    try:
        with path.open("rb") as f:
            if f.read(4) != b"GGUF":
                return False
            struct.unpack("<I", f.read(4))
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]

            def read_str() -> str:
                n = struct.unpack("<Q", f.read(8))[0]
                return f.read(n).decode("utf-8", "replace")

            def skip_value(vtype: int) -> None:
                if vtype in (0, 8):
                    f.read(1)
                elif vtype in (1, 9):
                    f.read(8)
                elif vtype in (2, 10):
                    f.read(4)
                elif vtype == 3:
                    f.read(2)
                elif vtype == 4:
                    f.read(4)
                elif vtype == 5:
                    f.read(4)
                elif vtype == 6:
                    f.read(1)
                elif vtype == 7:
                    read_str()
                elif vtype == 11:
                    f.read(8)
                elif vtype == 12:
                    elem = struct.unpack("<I", f.read(4))[0]
                    count = struct.unpack("<Q", f.read(8))[0]
                    for _ in range(count):
                        skip_value(elem)
                elif vtype == 13:
                    f.read(2)
                else:
                    raise ValueError(f"bad GGUF value type {vtype}")

            for _ in range(n_kv):
                key = read_str()
                skip_value(struct.unpack("<I", f.read(4))[0])
            for _ in range(n_tensors):
                name = read_str().lower()
                if "mtp." in name or name.startswith("predict.") or ".predict" in name:
                    return True
                n_dims = struct.unpack("<I", f.read(4))[0]
                f.read(8 * n_dims + 4 + 8)
    except Exception:
        return False
    return False


def _tensor_to_temp_png(tensor) -> Path:
    import numpy as np
    from PIL import Image

    array = (tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    fd, path = tempfile.mkstemp(prefix="llamacli-image-", suffix=".png")
    os.close(fd)
    Image.fromarray(array).save(path, format="PNG")
    return Path(path)


def image_to_temp_pngs(image) -> list[Path]:
    if hasattr(image, "dim") and image.dim() == 4:
        return [_tensor_to_temp_png(t) for t in image]
    return [_tensor_to_temp_png(image)]


def resolve_mmproj(model_dir: Path, mmproj: str) -> Path:
    if (mmproj or "").strip():
        path = Path(mmproj.strip()).expanduser()
        if not path.is_absolute():
            path = Path(folder_paths.models_dir) / path
    else:
        hits = sorted(model_dir.glob("*mmproj*.gguf"))
        if not hits:
            raise ValueError(f"Image input needs an mmproj next to {model_dir}, or set the mmproj input.")
        path = hits[0]
    if not path.is_file():
        raise ValueError(f"mmproj not found: {path}")
    return path


def _split_extra_args(extra_args: str) -> list[str]:
    if not extra_args or not extra_args.strip():
        return []
    return [part.strip("\"'") for part in shlex.split(extra_args, posix=(os.name != "nt"))]


def _write_temp(text: str, prefix: str, cleanup: list[Path]) -> str:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".txt")
    os.close(fd)
    cleanup.append(Path(path))
    Path(path).write_text(text, encoding="utf-8", newline="\n")
    return path


def build_command(cli: str, model_path: Path, args: dict) -> list[str]:
    cmd = [
        cli,
        "-m", str(model_path),
        "-n", str(args["max_length"]),
        "-c", str(args["ctx_size"]),
        "--seed", str(args["seed"]),
        "--no-display-prompt",
        "--no-escape",
    ]
    if args["chat"]:
        cmd += ["--reasoning", args["reasoning"], "--single-turn"]
        if args["system_prompt"]:
            if args["system_prompt_path"]:
                cmd += ["-sysf", args["system_prompt_path"]]
            else:
                cmd += ["-sys", args["system_prompt"]]
    if args["prompt_path"]:
        cmd += ["-f", args["prompt_path"]]
    else:
        cmd += ["-p", args["prompt"]]
    if args["sampling_on"]:
        cmd += [
            "--temp", str(args["temperature"]),
            "--top-k", str(args["top_k"]),
            "--top-p", str(args["top_p"]),
            "--min-p", str(args["min_p"]),
            "--repeat-penalty", str(args["repetition_penalty"]),
        ]
        if args["presence_penalty"] > 0:
            cmd += ["--presence-penalty", str(args["presence_penalty"])]
    else:
        cmd += ["--temp", "0", "--top-k", "0", "--top-p", "1", "--min-p", "0",
                "--repeat-penalty", "1", "--presence-penalty", "0"]
    if args["n_gpu_layers"] >= 0:
        cmd += ["-ngl", str(args["n_gpu_layers"])]
    if args["flash_attn"] != "auto":
        cmd += ["--flash-attn", args["flash_attn"]]
    if args["kv_cache_type"] != "f16":
        cmd += ["-ctk", args["kv_cache_type"], "-ctv", args["kv_cache_type"]]
    if args["threads"] >= 1:
        cmd += ["-t", str(args["threads"])]
    if args["batch_size"] > 0:
        cmd += ["-b", str(args["batch_size"])]
    if args["ubatch_size"] > 0:
        cmd += ["-ub", str(args["ubatch_size"])]
    if args["load_mode"] != "auto":
        cmd += ["--load-mode", args["load_mode"]]
    if args["fit"] == "off":
        cmd += ["--fit", "off"]
    if args["chat"]:
        if args["spec_type"]:
            cmd += ["--spec-type", args["spec_type"]]
            if args["spec_n_max"]:
                cmd += ["--spec-draft-n-max", str(args["spec_n_max"])]
        if args["mmproj_path"] is not None:
            cmd += ["--mmproj", str(args["mmproj_path"]), "--image", args["image_files"]]
    cmd += args["extra_args"]
    return cmd


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def _communicate_with_interrupt(process: subprocess.Popen, timeout_seconds: int) -> tuple[str, str]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        if comfy.model_management.processing_interrupted():
            _stop_process(process)
            comfy.model_management.throw_exception_if_processing_interrupted()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _stop_process(process)
            raise TimeoutError(f"llama-cli timed out after {timeout_seconds}s")
        try:
            stdout, stderr = process.communicate(timeout=min(0.1, remaining))
            return stdout, stderr
        except subprocess.TimeoutExpired:
            continue


def run_llama_cli(command: list[str], timeout_seconds: int, cleanup_paths: list[Path],
                  prompt: str = "", chat: bool = True) -> tuple[str, str, str]:
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
        )
    except OSError as exc:
        raise RuntimeError(f"cannot start llama-cli ({command[0]}): {exc}") from exc
    started = time.monotonic()
    try:
        stdout, stderr = _communicate_with_interrupt(process, timeout_seconds)
    finally:
        for path in cleanup_paths:
            if path and path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
    if process.returncode != 0:
        tail = (stderr or stdout or "").strip()
        raise RuntimeError(
            f"llama-cli exited with code {process.returncode}:\n"
            f"{'\n'.join(tail.splitlines()[-15:])}"
        )
    return parse_response(stdout, stderr, time.monotonic() - started, prompt, chat)


def parse_response(stdout: str, stderr: str, wall: float, prompt: str = "", chat: bool = True) -> tuple[str, str, str]:
    stdout = stdout or ""
    match = PERF_RE.search(stdout) or PERF_RE.search(stderr or "")
    content = stdout[: match.start()] if (match and match.string is stdout) else stdout

    # llama-cli conversation mode prints a banner and echoes "> {prompt}" before the answer.
    if chat and prompt.strip():
        stripped = prompt.strip()
        first_line = stripped.splitlines()[0]
        marker = "\n> " + first_line
        idx = content.find(marker)
        if idx != -1:
            content = content[idx + len(marker):]
            rest = stripped.splitlines()[1:]
            if rest:
                continuation = "\n" + "\n".join(rest)
                if content.startswith(continuation):
                    content = content[len(continuation):]
    content = content.strip()

    perf = match.group(0).strip() if match else ""
    stats = f"{perf} | node wall {wall:.1f}s" if perf else f"node wall {wall:.1f}s"

    # Comfy TE splits think blocks; llama.cpp may emit that or its own markers.
    thinking = ""
    if content.startswith(START_THINKING):
        thinking, _, content = content[len(START_THINKING):].partition(END_THINKING)
    else:
        head, sep, tail = content.partition("</think>")
        if sep and head.lstrip().startswith("<think>"):
            thinking = head.lstrip()[len("<think>"):]
            content = tail
        elif content.startswith(START_REDACTED):
            # --reasoning forced on a chat template that only reports a redacted trace
            thinking, _, content = content[len(START_REDACTED):].partition(END_REDACTED)
    for marker in RAW_EOF_MARKERS:
        idx = content.find(marker)
        if idx != -1:
            content = content[:idx]
    return content.strip(), thinking.strip(), stats


class LlamaCliTextGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        sampling_options = [
            io.DynamicCombo.Option(
                key="on",
                inputs=[
                    io.Float.Input("temperature", default=0.7, min=0.01, max=2.0, step=0.000001),
                    io.Int.Input("top_k", default=64, min=0, max=1000),
                    io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01),
                    io.Float.Input("min_p", default=0.05, min=0.0, max=1.0, step=0.01),
                    io.Float.Input("repetition_penalty", default=1.05, min=0.0, max=5.0, step=0.01),
                    io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                    io.Float.Input("presence_penalty", optional=True, default=0.0, min=0.0, max=5.0, step=0.01),
                ]
            ),
            io.DynamicCombo.Option(
                key="off",
                inputs=[]
            ),
        ]

        return io.Schema(
            node_id="LlamaCliTextGenerate",
            display_name="Generate Text (llama.cpp)",
            category="text",
            search_aliases=["LLM", "llama", "gguf", "prompt enhance"],
            inputs=[
                io.Combo.Input("model", options=list_gguf_models() or [""],
                               tooltip="Scans ComfyUI model folders (LLM, clip, text_encoders, diffusion_models, checkpoints, ...) for *.gguf. "
                                       "For files outside them use model_path_override. List refreshes within ~30s after F5."),
                io.String.Input("model_path_override", default="", advanced=True, force_input=True,
                                tooltip="Absolute path (or relative to the ComfyUI models dir) to a GGUF outside the scanned folders. Wins over model."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Image.Input("image", optional=True),
                io.String.Input("mmproj", default="", advanced=True,
                                tooltip="mmproj GGUF path for image input. Empty = auto-match *mmproj*.gguf next to the model."),
                io.Int.Input("max_length", default=512, min=1, max=32768,
                             tooltip="Max new tokens (-n). Generation stops early on EOS."),
                io.DynamicCombo.Input("sampling_mode", options=sampling_options, display_name="Sampling Mode"),
                io.Boolean.Input("thinking", optional=True, default=False,
                                 tooltip="--reasoning on/off for chat templates that support thinking."),
                io.Boolean.Input("use_default_template", optional=True, default=True, advanced=True,
                                 tooltip="Off = raw completion via llama-completion.exe (mirrors skip_template); system_prompt/image ignored."),
                io.String.Input("system_prompt", multiline=True, force_input=True, optional=True,
                                tooltip="Replaces the chat template system prompt (-sys). Empty = model default."),
                io.Combo.Input("mtp", options=["auto", "off", "2", "3", "4", "5"], default="auto", optional=True, advanced=True,
                               tooltip="MTP speculative decoding. auto only enables it when the GGUF contains MTP tensors."),
                io.Int.Input("ctx_size", default=16384, min=0, max=1048576, advanced=True,
                             tooltip="-c. 0 = model trained context. KV memory scales linearly."),
                io.Int.Input("n_gpu_layers", default=-1, min=-1, max=1024, advanced=True,
                             tooltip="-ngl. -1 = llama auto (all layers that fit), 0 = CPU only."),
                io.Combo.Input("flash_attn", options=["auto", "on", "off"], default="auto", advanced=True,
                               tooltip="-fa. on lowers KV memory."),
                io.Combo.Input("kv_cache_type", options=["f16", "q8_0", "bf16", "f32"], default="f16", advanced=True,
                               tooltip="-ctk/-ctv. q8_0 roughly halves KV cache VRAM."),
                io.Int.Input("threads", default=-1, min=-1, max=256, advanced=True,
                             tooltip="-t during generation. -1 = llama auto."),
                io.Int.Input("batch_size", default=2048, min=0, max=16384, advanced=True,
                             tooltip="-b logical batch for prompt processing. 0 = llama default."),
                io.Int.Input("ubatch_size", default=512, min=0, max=16384, advanced=True,
                             tooltip="-ub physical batch. Bigger speeds up prefill, costs VRAM. 0 = llama default."),
                io.Combo.Input("load_mode", options=["auto", "mmap", "mlock", "mmap+mlock", "none", "dio"], default="auto", advanced=True,
                               tooltip="--load-mode. mmap+mlock keeps the GGUF pinned in RAM: helps repeated loads on machines with little free RAM."),
                io.Combo.Input("fit", options=["on", "off"], default="on", advanced=True,
                               tooltip="--fit. on shrinks ctx/layers so llama.cpp starts even while other models hold VRAM."),
                io.String.Input("llama_cli_path", default="", advanced=True,
                                tooltip="llama.cpp bin dir or llama-cli.exe. Empty = auto: env LLAMA_CLI, llama_path.txt, PATH, vendor folders, drive-root llama* dirs."),
                io.String.Input("extra_args", default="", advanced=True,
                                tooltip="Raw llama-cli args appended last (shlex), can override anything above."),
                io.Int.Input("timeout", default=600, min=5, max=86400, advanced=True,
                             tooltip="Hard kill after this many seconds, ComfyUI interrupt also kills the child process."),
            ],
            outputs=[
                io.String.Output(display_name="generated_text"),
                io.String.Output(display_name="thinking"),
                io.String.Output(display_name="stats", tooltip="llama.cpp [Prompt|Generation] perf line + node wall time."),
            ],
        )

    @classmethod
    def execute(cls, model, prompt, max_length, sampling_mode,
                model_path_override="", image=None, mmproj="", system_prompt="",
                thinking=False, use_default_template=True, mtp="auto",
                ctx_size=16384, n_gpu_layers=-1, flash_attn="auto", kv_cache_type="f16",
                threads=-1, batch_size=2048, ubatch_size=512, load_mode="auto", fit="on",
                llama_cli_path="", extra_args="", timeout=600) -> io.NodeOutput:

        model_path = resolve_model_path(model, model_path_override)
        chat = bool(use_default_template)
        cli = find_llama_cli(llama_cli_path)
        if not chat:
            # llama-cli --no-conversation hangs in current builds; raw completion goes through llama-completion
            cli = find_sibling_exe(cli, "llama-completion")
            if image is not None:
                raise ValueError("Image input needs chat mode (use_default_template = on).")
        if not prompt.strip():
            raise ValueError("prompt is empty; llama-cli would wait for interactive input.")

        cleanup: list[Path] = []
        prompt_text = prompt.strip()
        prompt_path = ""
        if len(prompt_text) > 16000:
            prompt_path = _write_temp(prompt_text, "llamacli-prompt-", cleanup)
        system_prompt = (system_prompt or "").strip()
        system_path = ""
        if chat and len(system_prompt) > 16000:
            system_path = _write_temp(system_prompt, "llamacli-system-", cleanup)

        sampling_mode = sampling_mode or {}
        args = {
            "max_length": int(max_length),
            "ctx_size": int(ctx_size),
            "seed": normalize_llama_seed(sampling_mode.get("seed", 0) or 0),
            "chat": chat,
            "prompt": prompt_text,
            "prompt_path": prompt_path,
            "reasoning": "on" if thinking else "off",
            "system_prompt": system_prompt,
            "system_prompt_path": system_path,
            "sampling_on": sampling_mode.get("sampling_mode", "on") == "on",
            "temperature": sampling_mode.get("temperature", 0.7),
            "top_k": sampling_mode.get("top_k", 64),
            "top_p": sampling_mode.get("top_p", 0.95),
            "min_p": sampling_mode.get("min_p", 0.05),
            "repetition_penalty": sampling_mode.get("repetition_penalty", 1.05),
            "presence_penalty": sampling_mode.get("presence_penalty", 0.0) or 0.0,
            "n_gpu_layers": int(n_gpu_layers),
            "flash_attn": flash_attn,
            "kv_cache_type": kv_cache_type,
            "threads": int(threads),
            "batch_size": int(batch_size),
            "ubatch_size": int(ubatch_size),
            "load_mode": load_mode,
            "fit": fit,
            "spec_type": "",
            "spec_n_max": 0,
            "mmproj_path": None,
            "image_files": "",
            "extra_args": _split_extra_args(extra_args),
        }

        if mtp == "auto":
            if gguf_has_mtp_head(model_path):
                args["spec_type"] = "draft-mtp"
        elif mtp != "off":
            args["spec_type"] = "draft-mtp"
            args["spec_n_max"] = int(mtp)

        if chat and image is not None:
            pngs = image_to_temp_pngs(image)
            cleanup.extend(pngs)
            args["mmproj_path"] = resolve_mmproj(model_path.parent, mmproj)
            args["image_files"] = ",".join(str(p) for p in pngs)

        command = build_command(cli, model_path, args)
        text, reasoning, stats = run_llama_cli(command, int(timeout), cleanup, prompt_text, chat)
        return io.NodeOutput(text, reasoning, stats)


class LlamaCliExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LlamaCliTextGenerate,
        ]


async def comfy_entrypoint() -> LlamaCliExtension:
    return LlamaCliExtension()
