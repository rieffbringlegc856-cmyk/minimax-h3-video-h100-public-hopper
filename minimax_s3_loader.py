import io
import json
import av
import numpy as np
from PIL import Image
import requests
import torch


# -------------------------------------------------------------
# ЭМУЛЯЦИЯ / ДУБЛЕР НОДЫ: LayerUtility: ImageScaleByAspectRatio V2
# Вычисляет точные размеры кадра под формат (16:9, 9:16, 1:1, 4:3, 3:4)
# -------------------------------------------------------------
class LayerStyleScaleBypass:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "aspect_ratio": ("STRING", {"default": "16:9"}),
                "proportional_width": ("INT", {"default": 16}),
                "proportional_height": ("INT", {"default": 9}),
                "fit": ("STRING", {"default": "crop"}),
                "method": ("STRING", {"default": "lanczos"}),
                "round_to_multiple": ("STRING", {"default": "32"}),
                "scale_to_side": ("STRING", {"default": "longest"}),
                "scale_to_length": ("INT", {"default": 1280}),
                "background_color": ("STRING", {"default": "#000000"}),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "BOX", "width", "height")
    FUNCTION = "scale"
    CATEGORY = "LayerStyle/Bypass"

    def scale(self, image, aspect_ratio="16:9", scale_to_length=1280, **kwargs):
        scale_len = int(scale_to_length)
        ar = str(aspect_ratio).strip()

        if "9:16" in ar:
            h = scale_len
            w = int(round((scale_len * 9.0 / 16.0) / 32.0) * 32)
        elif "4:3" in ar:
            w = scale_len
            h = int(round((scale_len * 3.0 / 4.0) / 32.0) * 32)
        elif "3:4" in ar:
            h = scale_len
            w = int(round((scale_len * 3.0 / 4.0) / 32.0) * 32)
        elif "1:1" in ar:
            w = scale_len
            h = scale_len
        else:
            w = scale_len
            h = int(round((scale_len * 9.0 / 16.0) / 32.0) * 32)

        print(f"[LayerStyle Scale] Формат: {ar}, Итоговое разрешение: {w}x{h}", flush=True)

        mask = torch.zeros((1, h, w), dtype=torch.float32)
        box = (0, 0, w, h)
        return (image, mask, box, w, h)


# -------------------------------------------------------------
# ЭМУЛЯЦИЯ / ДУБЛЕРЫ НОД: ComfyUI-Easy-Use
# -------------------------------------------------------------
class EasySeedBypass:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "get_seed"
    CATEGORY = "EasyUse/Seed"

    def get_seed(self, seed):
        return (seed,)


class EasyShowAnythingBypass:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "text": ("STRING", {"forceInput": False, "default": ""}),
                "anything": ("*",),
            },
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("output", "text")
    FUNCTION = "show"
    CATEGORY = "EasyUse/Logic"

    def show(self, text="", anything=None, **kwargs):
        return (anything, str(text))


class EasyCleanGpuBypass:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "anything": ("*",),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "clean"
    CATEGORY = "EasyUse/Fix"

    def clean(self, anything=None, **kwargs):
        return (anything,)


class EasyIndexAnythingBypass:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 9999}),
            },
            "optional": {
                "any": ("*",),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "get_index"
    CATEGORY = "EasyUse/Logic"

    def get_index(self, index=0, any=None, **kwargs):
        if isinstance(any, (list, tuple)):
            if 0 <= index < len(any):
                return (any[index],)
            return (any[0] if len(any) > 0 else None,)
        return (any,)


# -------------------------------------------------------------
# ЗАГРУЗЧИК ИЗ YANDEX S3 И СПЛИТТЕР РЕФЕРЕНСОВ
# -------------------------------------------------------------
class MiniMax_S3_BatchLoader:
    MAX_VIDEO_SIDE = 768
    TARGET_VIDEO_FPS = 24.0
    MAX_VIDEO_FRAMES = 241  # ~10 секунд при 24 fps

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image_urls": ("STRING", {"multiline": True, "default": "[]"}),
                "video_urls": ("STRING", {"multiline": True, "default": "[]"}),
                "audio_urls": ("STRING", {"multiline": True, "default": "[]"}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("references",)
    FUNCTION = "load_media"
    CATEGORY = "MiniMax/Serverless"

    def _download_bytes(self, url):
        print(f"[MiniMax_S3] Скачивание: {url}", flush=True)
        res = requests.get(url.strip(), timeout=60)
        res.raise_for_status()
        return io.BytesIO(res.content)

    def _load_image(self, url):
        buf = self._download_bytes(url)
        img = Image.open(buf).convert("RGB")
        img_np = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(img_np)[None, ...]

    def _decode_audio_from_container(self, container):
        """Универсальное декодирование аудиопотока через PyAV в формат ComfyUI AUDIO."""
        if len(container.streams.audio) == 0:
            return None

        audio_stream = container.streams.audio[0]
        target_rate = audio_stream.rate or 44100
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=target_rate)

        chunks = []
        for frame in container.decode(audio=0):
            resampled_frames = resampler.resample(frame)
            for rf in resampled_frames:
                arr = rf.to_ndarray()  # shape: (channels, samples), float32
                chunks.append(arr)

        # Сбрасываем буфер ресемплера
        for rf in resampler.resample(None):
            chunks.append(rf.to_ndarray())

        if not chunks:
            return None

        waveform_np = np.concatenate(chunks, axis=1)  # (channels, total_samples)
        waveform = torch.from_numpy(waveform_np).unsqueeze(0).float()  # (1, channels, samples)
        return {"waveform": waveform, "sample_rate": int(target_rate)}

    def _resize_pil_frame(self, pil_img):
        w, h = pil_img.size
        longest = max(w, h)
        if longest > self.MAX_VIDEO_SIDE:
            scale = self.MAX_VIDEO_SIDE / float(longest)
            new_w = max(2, int(round((w * scale) / 2.0) * 2))
            new_h = max(2, int(round((h * scale) / 2.0) * 2))
            return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        # Гарантируем чётность сторон
        new_w = max(2, (w // 2) * 2)
        new_h = max(2, (h // 2) * 2)
        if (new_w, new_h) != (w, h):
            return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return pil_img

    def _load_video(self, url):
        buf = self._download_bytes(url)
        container = av.open(buf)

        if len(container.streams.video) == 0:
            raise ValueError("В файле отсутствует видеопоток")

        v_stream = container.streams.video[0]
        src_fps = float(v_stream.average_rate) if v_stream.average_rate else self.TARGET_VIDEO_FPS
        step = max(1, int(round(src_fps / self.TARGET_VIDEO_FPS))) if src_fps > self.TARGET_VIDEO_FPS + 2 else 1

        frames = []
        for idx, frame in enumerate(container.decode(video=0)):
            if idx % step != 0:
                continue
            pil_img = self._resize_pil_frame(frame.to_image().convert("RGB"))
            img_np = np.array(pil_img, dtype=np.float32) / 255.0
            frames.append(img_np)
            if len(frames) >= self.MAX_VIDEO_FRAMES:
                break

        if not frames:
            raise ValueError("Не удалось декодировать ни одного кадра из видео")

        video_tensor = torch.from_numpy(np.stack(frames))  # (F, H, W, C)
        print(
            f"[MiniMax_S3] Видео декодировано: {video_tensor.shape[0]} кадров, "
            f"разрешение {video_tensor.shape[2]}x{video_tensor.shape[1]} (исх. ~{src_fps:.1f} fps)",
            flush=True,
        )

        # Извлекаем аудиодорожку из того же буфера
        audio_dict = None
        if len(container.streams.audio) > 0:
            buf.seek(0)
            with av.open(buf) as audio_container:
                audio_dict = self._decode_audio_from_container(audio_container)
                if audio_dict:
                    print(
                        f"[MiniMax_S3] Аудио из видео извлечено: "
                        f"{audio_dict['waveform'].shape[-1]} сэмплов @ {audio_dict['sample_rate']} Hz",
                        flush=True,
                    )

        container.close()
        return video_tensor, audio_dict

    def _load_audio(self, url):
        buf = self._download_bytes(url)
        with av.open(buf) as container:
            audio_dict = self._decode_audio_from_container(container)
        if not audio_dict:
            raise ValueError("Не удалось декодировать аудиопоток из файла")
        print(
            f"[MiniMax_S3] Аудио декодировано: "
            f"{audio_dict['waveform'].shape[-1]} сэмплов @ {audio_dict['sample_rate']} Hz",
            flush=True,
        )
        return audio_dict

    def load_media(self, image_urls="[]", video_urls="[]", audio_urls="[]", **kwargs):
        def parse_urls(raw):
            raw = str(raw).strip()
            if not raw or raw == "[]":
                return []
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                return [
                    line.strip()
                    for line in raw.split("\n")
                    if line.strip().startswith("http")
                ]

        img_list = parse_urls(image_urls)[:9]
        vid_list = parse_urls(video_urls)[:3]
        aud_list = parse_urls(audio_urls)[:3]

        print(
            f"=== [MiniMax_S3] Запрошено: картинок={len(img_list)}, "
            f"видео={len(vid_list)}, аудио={len(aud_list)} ===",
            flush=True,
        )

        refs = {"pictures": [], "videos": [], "video_audios": [], "audios": []}

        for url in img_list:
            try:
                refs["pictures"].append(self._load_image(url))
            except Exception as e:
                print(f"[MiniMax_S3] Ошибка скачивания картинки {url}: {e}", flush=True)

        for url in vid_list:
            try:
                v_tensor, v_audio = self._load_video(url)
                refs["videos"].append(v_tensor)
                # Сохраняем None если у видео нет звука, чтобы не сбить индексы слотов
                refs["video_audios"].append(v_audio)
            except Exception as e:
                print(f"[MiniMax_S3] Ошибка скачивания видео {url}: {e}", flush=True)

        for url in aud_list:
            try:
                refs["audios"].append(self._load_audio(url))
            except Exception as e:
                print(f"[MiniMax_S3] Ошибка скачивания аудио {url}: {e}", flush=True)

        print(
            f"=== [MiniMax_S3] Загружено в память: картинок={len(refs['pictures'])}, "
            f"видео={len(refs['videos'])}, аудио_из_видео={sum(1 for a in refs['video_audios'] if a is not None)}, "
            f"отдельных_аудио={len(refs['audios'])} ===",
            flush=True,
        )
        return (refs,)


class MiniMaxH3ReferenceSplitter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"references": ("*",)}}

    RETURN_NAMES = (
        "picture_1", "picture_2", "picture_3", "picture_4", "picture_5",
        "picture_6", "picture_7", "picture_8", "picture_9",
        "video_1", "video_2", "video_3",
        "video_audio_1", "video_audio_2", "video_audio_3",
        "audio_1", "audio_2", "audio_3",
        "pictures", "videos", "audios",
    )
    RETURN_TYPES = tuple(["*"] * len(RETURN_NAMES))
    FUNCTION = "split"
    CATEGORY = "MiniMax"

    def split(self, references=None):
        if not isinstance(references, dict):
            return tuple([None] * len(self.RETURN_NAMES))

        pics = references.get("pictures", [])
        vids = references.get("videos", [])
        v_auds = references.get("video_audios", [])
        auds = references.get("audios", [])

        out = []
        for i in range(9):
            out.append(pics[i] if i < len(pics) else None)
        for i in range(3):
            out.append(vids[i] if i < len(vids) else None)
        for i in range(3):
            out.append(v_auds[i] if i < len(v_auds) else None)
        for i in range(3):
            out.append(auds[i] if i < len(auds) else None)
        out.extend([pics, vids, auds])

        return tuple(out)


class ImageExists:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "switch": ("*",),
                "image": ("*",),
            },
        }

    RETURN_TYPES = ("*", "BOOLEAN")
    RETURN_NAMES = ("image", "exists")
    FUNCTION = "check_exists"
    CATEGORY = "MiniMax"

    def check_exists(self, switch=None, image=None):
        val = switch if switch is not None else image

        if isinstance(val, (list, tuple)):
            val = val[0] if len(val) > 0 else None

        exists = False
        if val is not None and isinstance(val, torch.Tensor):
            if val.numel() > 64 * 64 * 3:
                exists = True
            elif val.numel() > 0 and (val.shape[-2] > 64 or val.shape[-3] > 64):
                exists = True

        if not exists or val is None:
            val = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        return (val, bool(exists))


# -------------------------------------------------------------
# РЕГИСТРАЦИЯ ВСЕХ НОД В COMFYUI
# -------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "MiniMax_S3_BatchLoader": MiniMax_S3_BatchLoader,
    "MiniMaxH3MediaLoader": MiniMax_S3_BatchLoader,
    "MiniMaxH3ReferenceSplitter": MiniMaxH3ReferenceSplitter,
    "ImageExists": ImageExists,
    "easy seed": EasySeedBypass,
    "easy showAnything": EasyShowAnythingBypass,
    "easy cleanGpuUsed": EasyCleanGpuBypass,
    "easy indexAnything": EasyIndexAnythingBypass,
    "LayerUtility: ImageScaleByAspectRatio V2": LayerStyleScaleBypass,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMax_S3_BatchLoader": "MiniMax S3 Batch Loader",
    "MiniMaxH3MediaLoader": "MiniMax H3 Media Loader",
    "MiniMaxH3ReferenceSplitter": "MiniMax H3 Reference Splitter",
    "ImageExists": "Image Exists",
    "easy seed": "Prompt Seed Bypass",
    "easy showAnything": "Show Anything Bypass",
    "easy cleanGpuUsed": "Clean VRAM Bypass",
    "easy indexAnything": "Index Anything Bypass",
    "LayerUtility: ImageScaleByAspectRatio V2": "LayerStyle Scale Bypass",
}
