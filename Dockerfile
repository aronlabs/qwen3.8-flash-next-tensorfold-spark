# TensorFold 0.6.1 + Concurrency & Speed Optimizations (v061-concurrent.patch)
ARG BASE_IMAGE=tensorfold-qwen38:v0.6.0
FROM ${BASE_IMAGE}
ARG PATCH_HASH=unset

# Upgrade TensorFold to upstream v0.6.1
RUN pip install --no-cache-dir --upgrade "git+https://github.com/ashhart/TensorFold.git@v0.6.1"
RUN tensorfold --version

WORKDIR /usr/local/lib/python3.12/dist-packages
RUN rm -f tensorfold/vision/videos.py tensorfold/families/qwen4_exp/cuda/ssd_read.cpp \
          tensorfold/families/qwen4_exp/cuda/ssd_read.py
COPY v061-concurrent.patch /opt/tf-patches/v061-concurrent.patch
RUN patch -p0 --forward --no-backup-if-mismatch < /opt/tf-patches/v061-concurrent.patch \
 && python -m compileall -q tensorfold \
 && python -c "import tensorfold.families.qwen4_exp.cuda.multi" \
 && python -c "import tensorfold.families.qwen4_exp.cuda.engine" \
 && python -c "import tensorfold.families.qwen4_exp.cuda.attn_multi" \
 && python -c "import tensorfold.families.qwen4_exp.cuda.gdn_multi" \
 && python -c "import tensorfold.families.qwen4_exp.cuda.ssd_read"
LABEL tf.version="0.6.1"
LABEL tf.patches="${PATCH_HASH}"

ENV HF_HOME=/root/.cache/huggingface \
    TORCH_EXTENSIONS_DIR=/cache/torch_extensions_v061 \
    TRITON_CACHE_DIR=/cache/triton_v061

WORKDIR /workspace
