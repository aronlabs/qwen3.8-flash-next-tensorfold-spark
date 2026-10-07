FROM tensorfold-qwen38:v0.6.0

# TensorFold python-0.6 branch at ed78d6f: v0.6.6 + 12 commits, incl. Flash Next CUDA copy drafts (PR #468)
RUN pip install --no-cache-dir --upgrade git+https://github.com/ashhart/TensorFold.git@ed78d6fc204d89d90b045bf033d6551e7714f3a1
RUN tensorfold --version

WORKDIR /usr/local/lib/python3.12/dist-packages
COPY v065-concurrent-vision.patch /opt/tf-patches/v065-concurrent-vision.patch
COPY local-copy-match.patch /opt/tf-patches/local-copy-match.patch
COPY src/tensorfold/families/qwen4_exp/cuda/ssd_read.cpp tensorfold/families/qwen4_exp/cuda/
COPY src/tensorfold/families/qwen4_exp/cuda/ssd_read.py tensorfold/families/qwen4_exp/cuda/

RUN patch -p2 --forward --no-backup-if-mismatch < /opt/tf-patches/v065-concurrent-vision.patch  && python -m compileall -q tensorfold  && python -c 'import tensorfold.families.qwen4_exp.cuda.multi'  && python -c 'import tensorfold.families.qwen4_exp.cuda.engine'  && python -c 'import tensorfold.families.qwen4_exp.cuda.attn_multi'  && python -c 'import tensorfold.families.qwen4_exp.cuda.gdn_multi'  && python -c 'import tensorfold.families.qwen4_exp.cuda.ssd_read'

# Copy drafts at --mtp-drafts 6: match 8 tokens, trim the copied chain to the depth
RUN patch -p2 --forward --no-backup-if-mismatch < /opt/tf-patches/local-copy-match.patch \
 && python -m compileall -q tensorfold/families/qwen4_exp/cuda/copy_drafts.py \
 && python -c 'from tensorfold.families.qwen4_exp.cuda.copy_drafts import CopyIndex as C; c=C(list(range(20))+list(range(10))); assert c.chain(6)==[10,11,12,13,14,15], c.chain(6); assert c.chain(12)==list(range(10,20))+[0,1], c.chain(12); assert C(list(range(20))+list(range(6))).chain(6)==[], "needs 8-token tail"'

LABEL tf.version=0.6.6-py06-ed78d6f-copy8
ENV HF_HOME=/root/.cache/huggingface     TORCH_EXTENSIONS_DIR=/cache/torch_extensions_v065     TRITON_CACHE_DIR=/cache/triton_v065
WORKDIR /workspace
