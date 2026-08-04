#!/bin/bash

# Ascend NPU 多卡容器启动脚本（精简版：无锁卡/隔离）
# 用法: bash start_docker_simple.sh <容器名> <芯片ID> <端口>
# 芯片ID支持: 单卡(0)、逗号分隔(0,1,2,3)、范围(0-3)、混合(0,2,4-7)

if [ -z "$1" ]; then
    read -e -p "请输入容器名: " NAME
    [ -z "$NAME" ] && { echo "Error: 容器名不能为空" >&2; exit 1; }
else
    NAME=$1
fi

if [ -z "$2" ]; then
    read -e -p "请输入芯片ID (如 0 或 0,1,2,3 或 0-3): " CHIPS_SPEC
    [ -z "$CHIPS_SPEC" ] && { echo "Error: 芯片ID不能为空" >&2; exit 1; }
else
    CHIPS_SPEC=$2
fi

if [ -z "$3" ]; then
    read -e -p "请输入端口号 (默认8001): " PORT
    PORT=${PORT:-8001}
else
    PORT=$3
fi

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# === 选择镜像 ===
select_image() {
    local default_image="quay.io/ascend/vllm-ascend:v0.19.1rc1-a3"
    local -a img_list=()
    local -a img_source=()

    echo ""
    echo "--- 选择 Docker 镜像 ---"
    read -e -p "输入搜索关键词 (回车列出全部): " kw

    # 已加载的本地镜像
    local loaded
    loaded=$(docker images --format "{{.Repository}}:{{.Tag}}" 2>/dev/null | grep -v "<none>" | sort -u)
    [ -n "$kw" ] && loaded=$(echo "$loaded" | grep -i "$kw")
    while IFS= read -r img; do
        [ -z "$img" ] && continue
        img_list+=("$img")
        img_source+=("local")
    done <<< "$loaded"

    # docker_create 目录下的 .tar / .tar.gz 文件
    local tars
    tars=$(find "${SCRIPT_DIR}" -maxdepth 1 \( -name "*.tar" -o -name "*.tar.gz" \) 2>/dev/null | sort)
    [ -n "$kw" ] && tars=$(echo "$tars" | grep -i "$kw")
    while IFS= read -r tar; do
        [ -z "$tar" ] && continue
        img_list+=("$(basename "$tar") [未加载]")
        img_source+=("$tar")
    done <<< "$tars"

    if [ ${#img_list[@]} -eq 0 ]; then
        echo "未找到匹配镜像，使用默认: ${default_image}"
        IMAGE="$default_image"
        return
    fi

    echo ""
    echo "可用镜像:"
    local i=1
    for item in "${img_list[@]}"; do
        printf "  [%2d] %s\n" "$i" "$item"
        i=$((i+1))
    done
    echo "  [0] 手动输入镜像名称"
    echo "  [回车] 使用默认: ${default_image}"
    echo ""

    read -e -p "请选择: " choice

    if [ -z "$choice" ]; then
        IMAGE="$default_image"
    elif [ "$choice" = "0" ]; then
        read -e -p "请输入镜像名称: " IMAGE
        IMAGE="${IMAGE:-$default_image}"
    elif [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#img_list[@]}" ]; then
        local idx=$((choice-1))
        local src="${img_source[$idx]}"
        if [ "$src" = "local" ]; then
            IMAGE="${img_list[$idx]}"
        else
            echo "加载镜像文件: ${src} (可能需要较长时间)..."
            local loaded_name
            loaded_name=$(docker load -i "$src" 2>&1 | grep "Loaded image:" | head -1 | sed 's/Loaded image:[[:space:]]*//')
            if [ -n "$loaded_name" ]; then
                IMAGE="$loaded_name"
            else
                echo "Warning: 无法自动获取镜像名，请手动输入"
                read -e -p "镜像名称: " IMAGE
                IMAGE="${IMAGE:-$default_image}"
            fi
        fi
    else
        IMAGE="$default_image"
    fi

    echo "使用镜像: ${IMAGE}"
    echo ""
}

select_image

# === 展开芯片ID规格: 支持 "0", "0,1,2", "0-3", "0,2,4-7" ===
expand_chips() {
    local spec="$1" result=""
    IFS=',' read -ra parts <<< "$spec"
    for part in "${parts[@]}"; do
        part=$(echo "$part" | tr -d ' ')
        if [[ "$part" == *-* ]]; then
            local s=${part%-*} e=${part#*-}
            for ((i=s; i<=e; i++)); do
                result="${result:+$result,}$i"
            done
        else
            result="${result:+$result,}$part"
        fi
    done
    echo "$result"
}

CHIPS_STR=$(expand_chips "$CHIPS_SPEC")
IFS=',' read -ra CHIPS <<< "$CHIPS_STR"

# === 验证芯片ID ===
for chip in "${CHIPS[@]}"; do
    if [[ ! "$chip" =~ ^[0-9]+$ ]] || [ "$chip" -lt 0 ] || [ "$chip" -gt 15 ]; then
        echo "Error: 无效芯片ID: $chip (范围 0-15)" >&2; exit 1
    fi
done
CHIP_COUNT=${#CHIPS[@]}

# === 构建 docker 参数 ===
DOCKER_RUN=(docker run -itd)
DOCKER_RUN+=(--name "${NAME}" --ipc=host --cap-add=ALL)

for chip in "${CHIPS[@]}"; do
    DOCKER_RUN+=(--device="/dev/davinci${chip}")
done

DOCKER_RUN+=(
    --device=/dev/davinci_manager
    --device=/dev/devmm_svm
    --device=/dev/hisi_hdc
    -v /usr/local/dcmi:/usr/local/dcmi
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware
    -v /usr/local/Ascend/ascend-toolkit:/usr/local/Ascend/ascend-toolkit
    -v /etc/ascend_install.info:/etc/ascend_install.info
    -v /sys:/sys
    -v /data:/data
    -v /home:/home
    -v /mnt:/mnt
    -v /opt/data/verification/models:/root/.cache
    -p ${PORT}:${PORT}
    -e ASCEND_RT_VISIBLE_DEVICES="${CHIPS_STR}"
    -e ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
    -e VLLM_USE_MODELSCOPE=True
    -e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
    -e LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64/plugin:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/inner:/usr/local/dcmi:/usr/local/lib:/usr/lib
    -e PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe
    -e PATH=/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/latest/tools/ccec_compiler/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    ${IMAGE}
    bash
)

# === 启动容器 ===
echo "启动容器 ${NAME} (${CHIP_COUNT} 卡: ${CHIPS_STR})..."
"${DOCKER_RUN[@]}"

if [ $? -eq 0 ]; then
    echo "芯片 ${CHIPS_STR} 已分配给容器 ${NAME}"
    echo ""
    echo "容器创建成功，进入容器请执行命令：docker exec -it ${NAME} bash"
    echo ""
else
    echo "Error: 容器启动失败" >&2; exit 1
fi
