FROM tensorfold-qwen38:v0.6.0

RUN pip install --no-cache-dir --upgrade git+https://github.com/ashhart/TensorFold.git@v0.6.5
RUN tensorfold --version

WORKDIR /usr/local/lib/python3.12/dist-packages
COPY v065-concurrent-vision.patch /opt/tf-patches/v065-concurrent-vision.patch
COPY src/tensorfold/families/qwen4_exp/cuda/ssd_read.cpp tensorfold/families/qwen4_exp/cuda/
COPY src/tensorfold/families/qwen4_exp/cuda/ssd_read.py tensorfold/families/qwen4_exp/cuda/

RUN patch -p2 --forward --no-backup-if-mismatch < /opt/tf-patches/v065-concurrent-vision.patch  && python -m compileall -q tensorfold  && python -c 'import tensorfold.families.qwen4_exp.cuda.multi'  && python -c 'import tensorfold.families.qwen4_exp.cuda.engine'  && python -c 'import tensorfold.families.qwen4_exp.cuda.attn_multi'  && python -c 'import tensorfold.families.qwen4_exp.cuda.gdn_multi'  && python -c 'import tensorfold.families.qwen4_exp.cuda.ssd_read'

LABEL tf.version=0.6.5
ENV HF_HOME=/root/.cache/huggingface     TORCH_EXTENSIONS_DIR=/cache/torch_extensions_v065     TRITON_CACHE_DIR=/cache/triton_v065
WORKDIR /workspace
