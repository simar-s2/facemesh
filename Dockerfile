# Official PyTorch image: Python, PyTorch, CUDA and cuDNN already set up
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app
# print training progress right away instead of holding it in a buffer
ENV PYTHONUNBUFFERED=1

# install the extra libraries first so Docker can reuse this layer when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY facemesh/ facemesh/
COPY scripts/ scripts/

# data and run folders are mounted from the host at run time, not baked into the image
ENTRYPOINT ["python", "-m", "facemesh.train"]
CMD ["--data", "/data/coma.npz", "--out_dir", "/runs/latest", "--amp"]
