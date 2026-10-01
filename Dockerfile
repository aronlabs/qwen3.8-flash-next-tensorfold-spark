# TensorFold 0.6.0 + the Flash Next speed patches ported from the recipe's 0.3.6.3 set (0002, 0003, 0006, 0007).
ARG BASE_IMAGE=tensorfold-qwen38:v0.6.0
FROM ${BASE_IMAGE}
ARG PATCH_HASH=unset
WORKDIR /usr/local/lib/python3.12/dist-packages
# leftovers of the 0.3.6.3 image under the pip upgrade (not in 0.6.0's RECORD); the patch creates ssd_read.* again
RUN rm -f tensorfold/vision/videos.py tensorfold/families/qwen4_exp/cuda/ssd_read.cpp \
          tensorfold/families/qwen4_exp/cuda/ssd_read.py
COPY v060-speed.patch /opt/tf-patches/v060-speed.patch
RUN patch -p0 --forward --no-backup-if-mismatch < /opt/tf-patches/v060-speed.patch \
 && python -m compileall -q tensorfold \
 && python -c "import tensorfold.families.qwen4_exp.cuda.multi, tensorfold.families.qwen4_exp.cuda.engine"
LABEL tf.patches="${PATCH_HASH}"
WORKDIR /workspace
