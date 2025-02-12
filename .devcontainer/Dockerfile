FROM python:3.11-slim

# Install git and other dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set up work directory
WORKDIR /workspace

# Install dependencies
RUN pip install --upgrade pip

RUN git clone https://github.com/christiando/gaussian-toolbox.git && \
    cd gaussian-toolbox && \
    pip install .
    
# Default command
CMD ["sleep", "infinity"]