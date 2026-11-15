import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import boto3
import runpod

COMFY_URL = "http://127.0.0.1:8188"
LOG_PATH = "/workspace/comfyui.log"
COMFY_OUTPUT_DIR = "/workspace/ComfyUI/output"

# Настройки Yandex Object Storage (берутся строго из Environment Variables в RunPod)
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", "https://storage.yandexcloud.net")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
S3_PUBLIC_DOMAIN = os.environ.get(
    "S3_PUBLIC_DOMAIN",
    f"{S3_ENDPOINT_URL.rstrip('/')}/{S3_BUCKET_NAME}" if S3_BUCKET_NAME else "",
)


def ensure_comfyui_ready():
    """Единая точка инициализации: симлинки моделей, переменные Hopper и запуск ComfyUI."""
    print("=== [INIT] Checking models symlink ===", flush=True)
    if os.path.exists("/runpod-volume/ComfyUI/models"):
        os.system("rm -rf /workspace/ComfyUI/models && ln -s /runpod-volume/ComfyUI/models /workspace/ComfyUI/models")
    elif os.path.exists("/runpod-volume/models"):
        os.system("rm -rf /workspace/ComfyUI/models && ln -s /runpod-volume/models /workspace/ComfyUI/models")
    else:
        print("=== [WARNING] Models directory not found in /runpod-volume! ===", flush=True)

    # Авто-патч KJNodes для Hopper (sm_90): исправление _cuda_archs и добавление заглушки per_channel_fp8
    try:
        kj_path = "/workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py"
        if os.path.exists(kj_path):
            with open(kj_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            modified = False
            if "_cuda_archs[0]" in content:
                content = content.replace("_cuda_archs[0]", '(_cuda_archs[0] if _cuda_archs else "sm90")')
                modified = True

            # Добавляем заглушку per_channel_fp8, если её нет в файле
            if "def per_channel_fp8" not in content:
                stub = "\ndef per_channel_fp8(v, tensor_layout=0, smooth_v=False):\n    return v, None, None\n"
                content = stub + content
                modified = True

            if modified:
                with open(kj_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print("[INIT] KJNodes runtime patches applied successfully!", flush=True)
    except Exception as e:
        print(f"[INIT] KJNodes patch warning: {e}", flush=True)

    # Переменные CUDA под H100 SXM / H200 (архитектура Hopper sm_90)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
    os.environ["CUDA_ARCH"] = "90"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Патч Kijai SageAttention (на случай проверки версии/архитектуры внутри ноды)
    try:
        path = "/workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                code = f.read()
            target = 'raise RuntimeError("sageattention is not new enough version or could not determine CUDA architecture, cannot apply MiniMax H3 Memory Efficient Sage Attention Patch.")'
            if target in code:
                code = code.replace(target, 'print("[KJNodes] SageAttention warning bypassed.")')
                with open(path, "w", encoding="utf-8") as f:
                    f.write(code)
                print("[INIT] KJNodes SageAttention check bypassed successfully", flush=True)
    except Exception as e:
        print(f"[INIT] Patch warning: {e}", flush=True)

    # Проверяем, не запущен ли уже ComfyUI
    try:
        req = urllib.request.Request(f"{COMFY_URL}/object_info/easy%20seed")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                print("=== [INIT] ComfyUI already running! ===", flush=True)
                return
    except Exception:
        pass

    print("=== [INIT] Starting ComfyUI process in background... ===", flush=True)
    log_file = open(LOG_PATH, "w", encoding="utf-8")
    proc = subprocess.Popen(
        ["python", "-u", "main.py", "--listen", "127.0.0.1", "--port", "8188", "--reserve-vram", "2.0"],
        cwd="/workspace/ComfyUI",
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    print(f"=== [INIT] Waiting for ComfyUI and custom nodes (PID: {proc.pid})... ===", flush=True)
    for _ in range(150):
        if proc.poll() is not None:
            print("=== [ERROR] ComfyUI process crashed unexpectedly! ===", flush=True)
            if os.path.exists(LOG_PATH):
                with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
                    print(f.read()[-3000:], flush=True)
            raise RuntimeError("ComfyUI crashed on startup")

        try:
            req = urllib.request.Request(f"{COMFY_URL}/object_info/easy%20seed")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    print("=== [INIT] ComfyUI and custom nodes successfully initialized! ===", flush=True)
                    return
        except Exception:
            pass
        time.sleep(2)

    raise TimeoutError("ComfyUI startup timed out after 300 seconds")


def upload_file_to_s3(local_path, s3_key):
    if not all([S3_BUCKET_NAME, S3_ACCESS_KEY, S3_SECRET_KEY]):
        return None, "Missing S3 environment variables (S3_BUCKET_NAME, S3_ACCESS_KEY, S3_SECRET_KEY)"

    try:
        session = boto3.session.Session()
        s3_client = session.client(
            service_name="s3",
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            region_name="ru-central1",
        )
        s3_client.upload_file(
            Filename=local_path,
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            ExtraArgs={"ContentType": "video/mp4"},
        )
        file_url = f"{S3_PUBLIC_DOMAIN.rstrip('/')}/{s3_key}"
        return file_url, None
    except Exception as e:
        err_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[ERROR] Failed to upload to Yandex S3: {err_msg}", flush=True)
        return None, err_msg


def get_comfy_logs(tail_lines=60):
    if not os.path.exists(LOG_PATH):
        return "Log file not found."
    try:
        with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            return "".join(lines[-tail_lines:])
    except Exception as e:
        return f"Error reading log: {e}"


def queue_prompt(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFY_URL}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        print("=== COMFYUI REJECTED WORKFLOW (400) ===", flush=True)
        print(error_body, flush=True)
        print("========================================", flush=True)
        raise RuntimeError(f"ComfyUI Validation Error: {error_body}")


def get_history(prompt_id):
    try:
        with urllib.request.urlopen(f"{COMFY_URL}/history/{prompt_id}") as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return {}


def _normalize_url_list(raw_val):
    if not raw_val:
        return []
    if isinstance(raw_val, list):
        return [str(x).strip() for x in raw_val if str(x).strip()]
    if isinstance(raw_val, str):
        s = raw_val.strip()
        if not s or s == "[]":
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
        return [s] if s.startswith("http") else []
    return []


def handler(event):
    job_input = event.get("input", {})

    with open("minimax_api_ready.json", "r", encoding="utf-8") as f:
        wf = json.load(f)

    # Нормализация путей Windows -> Linux
    for _, node_data in wf.items():
        if "inputs" in node_data:
            for k, v in node_data["inputs"].items():
                if isinstance(v, str) and "\\" in v:
                    node_data["inputs"][k] = v.replace("\\", "/")

    prompt_text = job_input.get("prompt", "A cinematic scene")
    duration = float(job_input.get("duration", 10.0))
    seed = int(job_input.get("seed", time.time()))
    main_steps = int(job_input.get("steps", 8))
    use_refiner = bool(job_input.get("use_refiner", False))
    refiner_steps = int(job_input.get("refiner_steps", 4))
    refiner_thick = int(job_input.get("refiner_thick", 20))
    sampler_name = job_input.get("sampler_name", "euler")
    scheduler = job_input.get("scheduler", "simple")
    aspect_ratio = job_input.get("aspect_ratio", "16:9")
    scale_to_length = int(job_input.get("longest_side") or job_input.get("scale_to_length") or 1280)

    image_urls = _normalize_url_list(job_input.get("image_urls"))
    video_urls = _normalize_url_list(job_input.get("video_urls"))
    audio_urls = _normalize_url_list(job_input.get("audio_urls"))

    has_any_refs = bool(image_urls or video_urls or audio_urls)
    has_audio_context = bool(video_urls or audio_urls or job_input.get("enable_audio", False))

    if "9430:8903" in wf:
        wf["9430:8903"]["inputs"]["string_b"] = prompt_text

    if "9438" in wf:
        wf["9438"]["inputs"]["Xi"] = duration
        wf["9438"]["inputs"]["Xf"] = duration

    if "9457" in wf:
        wf["9457"]["inputs"]["seed"] = seed
    if "9428" in wf:
        wf["9428"]["inputs"]["seed"] = seed

    if "9441:5775" in wf:
        wf["9441:5775"]["inputs"]["aspect_ratio"] = aspect_ratio
        wf["9441:5775"]["inputs"]["scale_to_length"] = scale_to_length

    if "9526" in wf:
        wf["9526"]["inputs"]["Xi"] = main_steps
        wf["9526"]["inputs"]["Xf"] = main_steps

    if "9593" in wf:
        wf["9593"]["inputs"]["value"] = use_refiner

    if "9581" in wf:
        wf["9581"]["inputs"]["Xi"] = refiner_steps
        wf["9581"]["inputs"]["Xf"] = refiner_steps

    if "9584" in wf:
        wf["9584"]["inputs"]["Xi"] = refiner_thick
        wf["9584"]["inputs"]["Xf"] = refiner_thick

    if "9522:9363" in wf:
        wf["9522:9363"]["inputs"]["item_0"] = sampler_name
        wf["9522:9363"]["inputs"]["item_1"] = scheduler

    # Передача всех типов референсов в S3 Batch Loader
    print(
        f"[HANDLER] Refs -> images: {len(image_urls)}, videos: {len(video_urls)}, audios: {len(audio_urls)}",
        flush=True,
    )
    if "9860" in wf:
        wf["9860"]["inputs"]["image_urls"] = json.dumps(image_urls)
        wf["9860"]["inputs"]["video_urls"] = json.dumps(video_urls)
        wf["9860"]["inputs"]["audio_urls"] = json.dumps(audio_urls)

    # Гарантируем выбор ноды MiniMaxH3ReferenceToVideo при наличии любых референсов (включая только аудио)
    for switch_id in ("9608", "9609"):
        if switch_id in wf and "inputs" in wf[switch_id]:
            wf[switch_id]["inputs"]["switch"] = has_any_refs

    # Настройка SolAttnPatch с поддержкой аудио-внимания
    sol_attn_defaults = {
        "sink_conditioning": "exact_kv",
        "start_percent": 0.0,
        "end_percent": 1.0,
        "min_tokens": 1024,
        "use_tma": False,
        "tau": 0.1,
        "morton": True,
        "morton_curve": "3d",
        "int8_pv": False,
        "int8_qk": False,
        "verbose": False,
        "dense_blocks": 0,
    }
    for _, n_data in wf.items():
        if n_data.get("class_type") == "SolAttnPatch":
            inputs = n_data.setdefault("inputs", {})
            for key, val in sol_attn_defaults.items():
                if key not in inputs or inputs[key] is True:
                    inputs[key] = val
            inputs["enable_audio"] = has_audio_context

    for _, n_data in wf.items():
        if n_data.get("class_type") == "DiffusionModelLoaderKJ":
            if "inputs" in n_data and "enable_fp16_accumulation" in n_data["inputs"]:
                n_data["inputs"]["enable_fp16_accumulation"] = False

    # Принудительно отключаем проблемный SolAttnPatch
    if "9859:9752" in wf and "inputs" in wf["9859:9752"]:
        wf["9859:9752"]["inputs"]["switch"] = False

    # Переключаем бэкенд внимания на стабильный comfy_kitchen_attention (обходит ошибки sageattention на Hopper)
    if "9859:9709" in wf:
        wf["9859:9709"]["inputs"]["choice"] = "comfy_kitchen_attention"
        wf["9859:9709"]["inputs"]["index"] = 0

   # Принудительно отключаем Mem Eff Sage Attention Patch, чтобы не вызывать сломанные бинарники _qattn_sm90
    if "9859:9752" in wf:
        wf["9859:9752"]["inputs"]["switch"] = False 

    # На H100 (80GB+) держим Qwen3-VL CLIP прямо на GPU для максимальной скорости
    if "9859:9853" in wf:
        wf["9859:9853"]["inputs"]["device"] = "default"

    # =========================================================================
    # ДИНАМИЧЕСКАЯ НАСТРОЙКА ЦЕПОЧКИ LORA (от 0 до 4 штук) с исправлением связей графа
    # =========================================================================
    raw_loras = job_input.get("loras", [])
    valid_loras = []
    for lora_item in raw_loras:
        l_name = str(lora_item.get("name") or lora_item.get("file") or "").strip()
        if l_name:
            if not l_name.endswith(".safetensors"):
                l_name += ".safetensors"
            try:
                l_str = float(lora_item.get("strength", 1.0))
            except (ValueError, TypeError):
                l_str = 1.0
            valid_loras.append({"name": l_name, "strength": l_str})

    print(f"[HANDLER] Активных LoRA к применению: {len(valid_loras)}", flush=True)

    # Исходная точка модели — выход ноды ModelSamplingMiniMaxH3 (нода 47)
    target_model_source = ["47", 0]
    last_lora_output_node = "47"

    for i in range(1, 5):
        node_id = f"9657:{i}"
        if node_id in wf:
            if i <= len(valid_loras):
                lora_data = valid_loras[i - 1]
                wf[node_id]["inputs"]["lora_name"] = lora_data["name"]
                wf[node_id]["inputs"]["strength_model"] = lora_data["strength"]
                
                if i == 1:
                    wf[node_id]["inputs"]["model"] = ["47", 0]
                else:
                    wf[node_id]["inputs"]["model"] = [f"9657:{i-1}", 0]
                
                last_lora_output_node = node_id
                print(f"  -> Слот {i} ({node_id}) активен: {lora_data['name']} (сила: {lora_data['strength']})", flush=True)
            else:
                del wf[node_id]
                print(f"  -> Слот {i} ({node_id}) отключен (нет лор)", flush=True)

    # Если лоры есть, точкой выхода становится последняя нода лор. Если лор нет, оставляем ноду 47.
    if valid_loras:
        target_model_source = [last_lora_output_node, 0]

    # Безопасно обновляем все ссылки на модель в воркфлоу, чтобы избежать KeyError
    for n_id, n_data in wf.items():
        if "inputs" in n_data:
            for k, v in n_data["inputs"].items():
                # Если узел ждет модель из старой логики лор или из любой ноды слотов 9657:*
                if isinstance(v, list) and len(v) == 2:
                    if v[0] in ("9440:9347", "9552:9550") or v[0].startswith("9657:"):
                        n_data["inputs"][k] = target_model_source

    # Порог чанков под H100 (убираем дробление под 24GB 4090)
    for n_id, n_data in wf.items():
        if n_data.get("class_type") == "MiniMaxChunkFeedForward" or n_id == "9859:9857":
            inputs = n_data.setdefault("inputs", {})
            inputs["chunks"] = 64
            inputs["seq_threshold"] = 2048

    queued = queue_prompt(wf)
    prompt_id = queued.get("prompt_id")
    if not prompt_id:
        return {"status": "error", "message": "Failed to queue workflow", "raw": queued}

    max_wait = 1800
    start_time = time.time()

    while time.time() - start_time < max_wait:
        history = get_history(prompt_id)
        if prompt_id in history:
            job_data = history[prompt_id]
            outputs = job_data.get("outputs", {})
            status_info = job_data.get("status", {})

            save_node_out = outputs.get("9546", {})
            raw_media_list = (
                save_node_out.get("videos", [])
                or save_node_out.get("gifs", [])
                or save_node_out.get("images", [])
            )

            uploaded_videos = []
            for item in raw_media_list:
                filename = item.get("filename")
                subfolder = item.get("subfolder", "")
                if filename:
                    local_filepath = os.path.join(COMFY_OUTPUT_DIR, subfolder, filename)
                    s3_key = f"outputs/{prompt_id}_{filename}"

                    if os.path.exists(local_filepath):
                        s3_url, upload_err = upload_file_to_s3(local_filepath, s3_key)
                        uploaded_videos.append(
                            {
                                "filename": filename,
                                "url": s3_url,
                                "s3_key": s3_key,
                                "upload_error": upload_err,
                            }
                        )
                    else:
                        uploaded_videos.append(
                            {
                                "filename": filename,
                                "error": f"File not found on disk: {local_filepath}",
                            }
                        )

            comfy_stdout = get_comfy_logs(80)

            return {
                "status": "success" if uploaded_videos else "incomplete_or_failed",
                "prompt_id": prompt_id,
                "execution_status": status_info,
                "outputs": outputs,
                "videos": uploaded_videos,
                "comfy_console_log": comfy_stdout,
            }
        time.sleep(2)

    return {
        "status": "timeout",
        "prompt_id": prompt_id,
        "comfy_console_log": get_comfy_logs(80),
    }


if __name__ == "__main__":
    ensure_comfyui_ready()
    runpod.serverless.start({"handler": handler})
