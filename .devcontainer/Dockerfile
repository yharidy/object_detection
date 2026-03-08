FROM python:3.11-bullseye


# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    nano \
    htop \
    libopencv-dev \
    python3-opencv \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    apt-transport-https \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK using the official method
RUN curl https://sdk.cloud.google.com | bash && \
    echo 'export PATH=/root/google-cloud-sdk/bin:$PATH' >> /root/.bashrc

# Create non-root user
RUN useradd -m -s /bin/bash vscode

# Set working directory
WORKDIR /workspace

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Make gcloud available to vscode user
RUN cp -r /root/google-cloud-sdk /home/vscode/google-cloud-sdk && \
    chown -R vscode:vscode /home/vscode/google-cloud-sdk && \
    echo 'export PATH=/home/vscode/google-cloud-sdk/bin:$PATH' >> /home/vscode/.bashrc

# Set proper permissions
RUN chown -R vscode:vscode /workspace

USER vscode

CMD ["/bin/bash"]
