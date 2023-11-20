FROM pytorch/pytorch:latest

# Install dependencies
ADD ./requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt