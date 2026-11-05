# Benchmark client image, built in Azure with ACR Tasks (python deploy.py bench-image) and run on Azure
# Container Instances in the endpoint's region (python deploy.py bench).
FROM mcr.microsoft.com/azurelinux/base/python:3.12
RUN (python3 -m pip --version || (tdnf install -y python3-pip && tdnf clean all)) \
 && python3 -m pip install --no-cache-dir aiohttp
WORKDIR /bench
COPY bench.py .
ENTRYPOINT ["python3", "bench.py"]
