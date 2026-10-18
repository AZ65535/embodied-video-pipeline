"""管线各步骤共用的基础件。

这里只放**跨步骤的契约与纯工具**，并且刻意只依赖标准库：

- 产物文件名、清单字段、帧文件名格式（生产者和消费者必须看到同一份定义）
- JPEG 头解析、模糊判定、视频输出名前缀这类被多处复用的纯函数
- JSON / JSONL 读写与日志初始化

不放任何一步的业务逻辑。第一步特有的解码（cv2）、第三步特有的 lerobot、
动作导入特有的 h5py 都留在各自脚本里 —— 本模块会被所有步骤 import，
一旦引入重依赖，第二步和辅助工具的「纯标准库」性质就没了。
"""

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, TypedDict


# ---- 第一步的产物（写进 output/images/）----
IMAGES_META_FILENAME = "_meta.json"
BLUR_SCORES_FILENAME = "_scores.jsonl"
# 输入目录里以下划线开头的文件都是上游的元数据，不是帧图片。
METADATA_PREFIX = "_"

# ---- 第二步的产物（写进 dataset/meta/，第三步读取）----
META_DIRNAME = "meta"
FRAMES_JSONL = "frames.jsonl"
EPISODES_JSONL = "episodes.jsonl"
TASKS_JSONL = "tasks.jsonl"
INFO_JSON = "info.json"

# ---- 数据集自描述与索引 ----
MANIFEST_FILENAME = "dataset_manifest.json"
INDEX_FILENAME = "datasets.json"
PIPELINE_VERSION = "0.1.0"
DATASET_MANIFEST_SCHEMA_VERSION = 1
INDEX_SCHEMA_VERSION = 1

# ---- 动作侧车（import_actions.py 写、export_lerobot.py 读）----
ACTIONS_DIRNAME = "actions"
ACTION_ARRAY = "action"  # npz 里的数组名
STATE_ARRAY = "state"
ACTION_KEY = "action"  # 数据集特征名
STATE_KEY = "observation.state"

# ---- 数据集特征里由本管线添加的列 ----
SOURCE_FRAME_KEY = "source_frame"
BLUR_SCORE_KEY = "blur_score"

DEFAULT_CAMERA = "front"
DEFAULT_BLUR_THRESHOLD = 100.0
DEFAULT_REPO_ID = "local/embodied-videos"

# 原始帧率的三种来源，会写进 info.json / README，方便判断 timestamp 可不可信。
FPS_ORIGIN_RECORDED = IMAGES_META_FILENAME  # 精确
FPS_ORIGIN_INFERRED = "inferred_from_frame_gaps"  # 近似，误差约 ±fps/2
FPS_ORIGIN_NOMINAL = "nominal"  # 兜底，时间轴会偏小

# 形如 "1_frame_000318.jpg"：视频名贪婪匹配，取最后一个 "_frame_" 作分隔，
# 这样视频名本身含下划线甚至含 "_frame_" 也能正确切分。
FRAME_NAME_RE = re.compile(
    r"^(?P<video>.+)_frame_(?P<frame>\d+)\.(?:jpg|jpeg)$", re.IGNORECASE
)

# JPEG SOF 标记：C0-CF 里除 C4(DHT)/C8(JPG)/CC(DAC) 之外都是帧头，携带宽高
SOF_MARKERS = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)
# 无长度字段的独立标记：TEM / RST0-7 / SOI / EOI
STANDALONE_MARKERS = frozenset(
    {0x01, 0xD8, 0xD9, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7}
)


class FrameRow(TypedDict):
    """format_dataset.py 写进 meta/frames.jsonl 的一行（一行一个相机帧）。"""

    index: int
    episode_index: int
    frame_index: int
    source_frame: int
    camera: str
    blur_score: Optional[float]
    is_blurry: Optional[bool]
    timestamp: float
    path: str


class EpisodeRow(TypedDict):
    """format_dataset.py 写进 meta/episodes.jsonl 的一行。"""

    episode_index: int
    tasks: List[str]
    length: int
    episode: str
    source_video: str
    cameras: List[str]
    source_videos: Dict[str, str]
    fps: int
    source_fps: Optional[float]
    source_fps_origin: Optional[str]
    source_fps_by_camera: Dict[str, Optional[float]]
    source_fps_origin_by_camera: Dict[str, Optional[str]]
    first_source_frame: int
    last_source_frame: int
    segment_index: int
    segment_count: int


def natural_key(text: str) -> List[object]:
    """自然排序键：让 "2" 排在 "10" 前面，而不是按字典序倒过来。"""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def frame_filename(video_name: str, source_frame: int) -> str:
    """第一步产出的帧文件名。格式定义在这里，解析和重建都走同一份规则。"""
    return f"{video_name}_frame_{source_frame:06d}.jpg"


def parse_frame_name(path: Path) -> Optional[Tuple[str, int]]:
    """从帧文件名解析出 (视频名, 原始帧号)；不匹配返回 None。"""
    match = FRAME_NAME_RE.match(path.name)
    if match is None:
        return None
    return match.group("video"), int(match.group("frame"))


def video_output_prefix(video_path: Path, input_dir: Path) -> str:
    """算出该视频的输出名前缀（图片文件名和 _meta.json 的键都用它）。

    规则：相对 input_dir 的路径去掉后缀，路径分隔符换成下划线。平铺目录下
    结果就是原来的 stem；子目录下带上目录名，避免同名视频互相覆盖。

    import_actions.py 也用它来把动作记录和帧清单对上 —— 两处必须共用同一份
    实现，否则表现是动作静默配错帧，不报错。
    """
    try:
        relative = video_path.relative_to(input_dir)
    except ValueError:
        return video_path.stem  # 不在输入目录下（理论上不会发生）
    return "_".join(relative.with_suffix("").parts)


def probe_jpeg(path: Path) -> Optional[Tuple[int, int]]:
    """只读 JPEG 头部，返回 (width, height)。

    顺带充当完整性校验：结构不对（缺 SOI、段长非法、SOS 前无 SOF）返回 None，
    调用方据此跳过并计入损坏帧。
    """
    try:
        with path.open("rb") as handle:
            if handle.read(2) != b"\xff\xd8":
                return None  # 缺 SOI

            while True:
                marker = handle.read(2)
                if len(marker) < 2 or marker[0] != 0xFF:
                    return None

                code = marker[1]
                # 允许 FF FF ... xx 形式的填充字节
                while code == 0xFF:
                    extra = handle.read(1)
                    if not extra:
                        return None
                    code = extra[0]

                if code == 0xDA or code == 0xD9:
                    return None  # 遇到 SOS/EOI 还没见到 SOF，说明不是常规 JPEG
                if code in STANDALONE_MARKERS:
                    continue

                length_bytes = handle.read(2)
                if len(length_bytes) < 2:
                    return None
                segment_length = int.from_bytes(length_bytes, "big")
                if segment_length < 2:
                    return None

                if code in SOF_MARKERS:
                    body = handle.read(5)  # 精度(1) + 高(2) + 宽(2)
                    if len(body) < 5:
                        return None
                    height = int.from_bytes(body[1:3], "big")
                    width = int.from_bytes(body[3:5], "big")
                    if width <= 0 or height <= 0:
                        return None
                    return width, height

                handle.seek(segment_length - 2, os.SEEK_CUR)
    except OSError:
        return None


def has_eoi(path: Path) -> bool:
    """检查文件末尾是否有 JPEG 结束标记 EOI（FFD9）。"""
    try:
        size = path.stat().st_size
        if size < 4:
            return False
        with path.open("rb") as handle:
            handle.seek(-2, os.SEEK_END)
            return handle.read(2) == b"\xff\xd9"
    except OSError:
        return False


def is_blurry(score: float, threshold: float) -> bool:
    """分数低于阈值判为模糊（严格小于：等于阈值算清晰）。

    这是全项目唯一一份阈值比较口径：第一步记录分数、第二步标 is_blurry、
    标定工具算过滤率，都调用这里。不要再在别处写 `score < threshold`。
    """
    return score < threshold


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def read_jsonl(path: Path, missing_hint: Optional[str] = None) -> List[dict]:
    """读一个 JSONL 文件；不存在或某行非法时抛 ValueError。

    missing_hint 用于文件缺失时补一句「该先跑哪一步」，让错误信息能直接行动。
    """
    if not path.is_file():
        hint = f"（{missing_hint}）" if missing_hint else ""
        raise ValueError(f"缺少文件：{path}{hint}")
    rows: List[dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON：{exc}") from exc
    return rows


def read_blur_scores(
    path: Path, logger: Optional[logging.Logger] = None
) -> Dict[str, Dict[int, float]]:
    """读第一步产出的 `_scores.jsonl`，返回 {视频名: {原始帧号: 分数}}。

    文件不存在时返回空字典，让调用方自己决定「缺失」是告警还是错误。
    坏行跳过并告警：上游旧版本可能没有这个文件，不该因此让整条管线停摆。
    """
    logger = logger or logging.getLogger(__name__)
    result: Dict[str, Dict[int, float]] = {}
    if not path.is_file():
        return result
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("%s 读取失败，忽略", path)
        return result

    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            score = row["blur_score"]
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError("blur_score 不是数字")
            result.setdefault(str(row["video"]), {})[int(row["source_frame"])] = float(score)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("跳过 %s 第 %d 行（%s）", path, lineno, exc)
    return result


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def atomic_write_json(
    path: Path, payload: dict, logger: Optional[logging.Logger] = None
) -> bool:
    """先写临时文件再替换，避免中途崩溃留下半截 JSON。成功返回 True。"""
    logger = logger or logging.getLogger(__name__)
    tmp_path = path.parent / (path.name + ".tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(tmp_path, path)
    except OSError:
        logger.exception("写入 %s 失败", path)
        return False
    return True


# ---- 多相机契约 ----

_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *{f"COM{i}" for i in range(1, 10)},
    *{f"LPT{i}" for i in range(1, 10)},
}
_INVALID_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')


def camera_key(name: str) -> str:
    return f"observation.images.{name}"


def validate_camera_name(name: str) -> Optional[str]:
    """返回 None 表示合法，否则返回原因。"""
    if not name:
        return "相机名不能为空"
    if name in {".", ".."}:
        return "相机名不能是 '.' 或 '..'"
    if len(name) > 64:
        return "相机名最长 64 个字符"
    if _INVALID_NAME_CHARS.search(name):
        return "相机名包含路径分隔符或 Windows 非法字符"
    if name[-1] in " .":
        return "相机名不能以空格或点结尾"
    if name.split(".")[0].upper() in _WINDOWS_RESERVED_NAMES:
        return "相机名是 Windows 保留设备名"
    return None


def canonical_camera_order(names: Iterable[str]) -> List[str]:
    """相机规范顺序：大小写不敏感的自然排序。"""
    return sorted(names, key=lambda name: natural_key(name.casefold()))


def load_camera_names(path: Optional[Path]) -> Dict[str, str]:
    """读取 --camera-names 映射（filename stem -> camera name）。"""
    if path is None:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取相机名映射 {path}：{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"相机名映射 {path} 必须是 JSON 对象")
    result: Dict[str, str] = {}
    for stem, name in payload.items():
        if not isinstance(stem, str) or not isinstance(name, str):
            raise ValueError(f"相机名映射 {path} 的键和值都必须是字符串")
        reason = validate_camera_name(name)
        if reason:
            raise ValueError(f"相机名映射 {path} 中 {stem!r} -> {name!r}：{reason}")
        result[stem] = name
    return result


def episode_key_for_dir(relative_dir: Path) -> str:
    """把相对 episode 目录转成文件系统安全的 episode key。

    根目录（`.`）使用 `_root`；路径分隔符换成 `_`；非法字符替换为 `_`；
    保留设备名加前缀；超长时截断并附加 sha1 前缀。
    """
    raw = "_".join(relative_dir.parts) if relative_dir.parts else "_root"
    if raw in {"", ".", "..", "_root"}:
        raw = "_root" if raw in {"", ".", ".."} else raw
    sanitized = _INVALID_NAME_CHARS.sub("_", raw).rstrip(" .")
    if not sanitized:
        sanitized = "_root"
    if sanitized.split(".")[0].upper() in _WINDOWS_RESERVED_NAMES:
        sanitized = "_" + sanitized
    if len(sanitized) > 120:
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        sanitized = sanitized[:100] + "_" + digest
    return sanitized
