FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="9.0" \
    CUDA_ARCH="90"

WORKDIR /workspace

# Системные пакеты (добавлен libsndfile1 для корректной работы аудио-кодеков)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libsndfile1 \
    git \
    curl \
    dos2unix \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Обновляем PyTorch под CUDA 12.4
RUN pip install --no-cache-dir --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Установка Triton и SageAttention (компилируются под Hopper 9.0)
RUN pip install --no-cache-dir --upgrade triton && \
    pip install --no-cache-dir sageattention

# Клонируем ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI

WORKDIR /workspace/ComfyUI
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir runpod requests av boto3 soundfile

# Клонируем кастомные ноды
WORKDIR /workspace/ComfyUI/custom_nodes
RUN git clone https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder.git && \
    git clone https://github.com/kijai/ComfyUI-SolAttn_triton.git && \
    git clone https://github.com/kijai/ComfyUI-KJNodes.git && \
    git clone https://github.com/rgthree/rgthree-comfy.git && \
    git clone https://github.com/yolain/ComfyUI-Easy-Use.git && \
    git clone https://github.com/theUpsider/ComfyUI-Logic.git && \
    git clone https://github.com/chflame163/ComfyUI_LayerStyle.git && \
    git clone https://github.com/Smirnov75/ComfyUI-mxToolkit.git && \
    git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git && \
    git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git && \
    git clone https://github.com/StableLlama/ComfyUI-basic_data_handling.git

# Устанавливаем зависимости нод
RUN for dir in */; do \
        if [ -f "${dir}requirements.txt" ]; then \
            pip install --no-cache-dir -r "${dir}requirements.txt"; \
        fi; \
    done

# Копируем и запускаем патч для comfy_kitchen после установки всех зависимостей
COPY patch_kitchen.py /workspace/patch_kitchen.py
RUN python3 /workspace/patch_kitchen.py

# Копируем наши скрипты и воркфлоу в самом конце для быстрого кеширования слоёв Docker
RUN mkdir -p /workspace/ComfyUI/custom_nodes/MiniMax_S3_Loader
COPY minimax_s3_loader.py /workspace/ComfyUI/custom_nodes/MiniMax_S3_Loader/__init__.py

WORKDIR /workspace/ComfyUI
COPY minimax_api_ready.json /workspace/ComfyUI/minimax_api_ready.json
COPY rp_handler.py /workspace/ComfyUI/rp_handler.py
COPY start.sh /workspace/ComfyUI/start.sh

RUN dos2unix /workspace/ComfyUI/start.sh && chmod +x /workspace/ComfyUI/start.sh

CMD ["/workspace/ComfyUI/start.sh"]
