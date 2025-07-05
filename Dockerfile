# setup
FROM python:3.11.5

WORKDIR /app
COPY requirements.txt /app
COPY *.py /app
COPY pyproject.toml /app

COPY src/ /app/src/
COPY examples/ /app/examples/

WORKDIR /app
RUN ls --recursive /app/
RUN pip3 install --upgrade -r requirements.txt
RUN python -m build .
RUN pip3 install .
RUN pip3 install gunicorn

# Install system dependencies including libGL for OpenCV
RUN apt clean
RUN apt-get update
RUN apt-get -y install build-essential libssl-dev ca-certificates libasound2 wget libgl1

# Setup OpenSSL manually (as required for some TTS services)
RUN wget -O - https://www.openssl.org/source/openssl-1.1.1w.tar.gz | tar zxf -
WORKDIR openssl-1.1.1w
RUN ./config --prefix=/usr/local
RUN make -j $(nproc)
RUN make install_sw install_ssldirs
RUN ldconfig -v
ENV SSL_CERT_DIR=/etc/ssl/certs

ENV PYTHONUNBUFFERED=1

WORKDIR /app

EXPOSE 8000
CMD ["gunicorn", "--workers=2", "--log-level", "debug", "--chdir", "examples/foundational", "--capture-output", "transcript:app", "--bind=0.0.0.0:8000"]

