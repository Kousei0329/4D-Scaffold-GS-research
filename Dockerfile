# STAC (4D-Scaffold-GS-research) 用コンテナ
# ホストの前提: NVIDIA Driver(CUDA 11.6以上対応) + nvidia-container-toolkit がインストール済みであること。
# ビルド元GPU: RTX 3090 x2 / Driver 570.158.01 / CUDA 11.6 toolkit を元に構成。
#
# build:  docker build -t stac:latest .
# run:    docker run --gpus all --shm-size=16g -it \
#           -v $(pwd):/workspace/4D-Scaffold-GS-research \
#           -v /path/to/N3DV:/workspace/data/N3DV \
#           stac:latest

FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget build-essential ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p "$CONDA_DIR" \
    && rm /tmp/miniconda.sh
ENV PATH="$CONDA_DIR/bin:$PATH"

# 新しいcondaはdefaultsチャンネル(pkgs/main, pkgs/r)のToS同意が無いと
# 非対話環境では `conda env create` が止まる(environment.ymlのdefaultsチャンネル利用のため)
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

WORKDIR /workspace/4D-Scaffold-GS-research

# 依存関係の変更が少ないファイルだけ先にCOPYしてレイヤーキャッシュを効かせる
COPY environment.yml ./
COPY submodules ./submodules

# docker build中はGPUが見えない(--gpusはrun時のオプションでbuildには効かない)ため、
# submodules配下のCUDA拡張をビルドするtorch.utils.cpp_extensionがtorch.cuda経由で
# アーキテクチャを自動検出できず `IndexError: list index out of range` で失敗する。
# 対象GPUのcompute capabilityを明示しておく必要がある(Ampere世代、RTX 3090=8.6)。
# 他のアーキテクチャのGPUで使う場合はここを調整(例: Turing追加なら "7.5;8.0;8.6+PTX")。
ENV TORCH_CUDA_ARCH_LIST="8.6+PTX"

RUN conda env create -f environment.yml && conda clean -afy

# 拡張ビルドの確認(ここで失敗したらsubmodules個別インストールへ)
RUN conda run -n 4d_scaffold_naitive python -c \
    "import diff_gaussian_rasterization, simple_knn, _gridencoder, arithmetic; print('all extensions OK')"

# 残りのソース一式をコピー(通常はこの行をbind mountで上書きして開発する)
COPY . .

ENV PATH="$CONDA_DIR/envs/4d_scaffold_naitive/bin:$PATH"
SHELL ["conda", "run", "-n", "4d_scaffold_naitive", "/bin/bash", "-c"]

CMD ["/bin/bash"]
