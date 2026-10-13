import base64
import json as js
import math
import uuid  # 生成会话唯一标识
from datetime import datetime
from io import BytesIO
from os import listdir, remove
from pathlib import Path
from zoneinfo import ZoneInfo

import openai
import streamlit as st
import streamlit.typing as sttype
from PIL import Image, UnidentifiedImageError

LOGO = "./chat_robot.png"
NEW_CHAT_ICON = ":material/add_comment:"
DELETE_CHAT_ICON = ":material/delete:"
SAMPLE_CHAT_PROMPT = "你能帮我做什么？"
BASE_URL = "http://172.20.10.4:11434/v1"
SYSTEM_PROMPT_TEMPLATE = """
你必须严格按以下顺序输出：
[[AGENT_META]]
{"title": null, "images": []}
[[/AGENT_META]]
正式回复正文...

规则：
1. AGENT_META 必须位于输出最开头，前面不能有任何文字
2. AGENT_META 中必须是合法 JSON 对象，不能使用 Markdown 代码块
3. 正式回复只能写在 [[/AGENT_META]] 之后
4. 不要向用户解释、复述或展示 AGENT_META
5. title 只有在系统明确要求总结标题时需要填写标题，其余情况保持 null；images 默认输出空数组，只有图片提示词中“本轮新上传图片”不为“无”时才能填写图片信息
6. 使用中文回答用户

用户对于你的要求：
你的名字：%s
你的性格：%s
"""

IMAGE_PROMPT = r"""
【历史图片目录，仅供摘要检索】
%s

【本轮精确历史图片】
%s

【本轮新上传图片】
%s

图片理解规则：
- “历史图片目录”只提供摘要和 OCR，供你概括理解历史图片
- “本轮精确历史图片”会附带原图，只供本轮精确回答
- “本轮新上传图片”会附带原图，是本轮唯一允许生成图片记忆卡的图片
- 某个图片若同时出现在目录和“精确历史图片”中，以附带的原图为准

AGENT_META.images 严格规则：
- 只能为“本轮新上传图片”列表中的 image_id 生成记忆卡
- 如果“本轮新上传图片”为“无”，必须输出 "images": []
- 历史图片目录和本轮精确历史图片中的任何图片，都禁止生成、更新或重复输出记忆卡
- 即使用户在本轮提及、询问、比较历史图片，也只可利用其摘要或附带原图回答，不能把历史图片写入 AGENT_META.images

图片记忆卡格式：
- 生成在 AGENT_META.images 项
- 每张格式：{"image_id": "本轮新上传图片ID", "summary": "图片摘要", "ocr_text": "关键文字"}
- summary：不超过 30 个汉字；概括图片主体、关键内容、可检索关键词和重要问题
- ocr_text：仅提取清晰且重要的可见文字；没有则使用空字符串
- 图片看不清或无法确认时，summary 中应说明不确定性；不要编造细节或 OCR 文字
- 不要在正式回复中展示 AGENT_META、JSON 或内部图片编号，除非用户主动要求图片编号
"""

AGENT_META_OPEN = "[[AGENT_META]]"
AGENT_META_CLOSE = "[[/AGENT_META]]"
TITLE_GENERATE_PROMPT = """
本轮是新会话的第一轮回复
请在 AGENT_META.title 项中生成一个标题

1.标题只概括用户问题主题
2.不要在标题中使用引号、标点、序号或换行
3. 不要在正式回复正文中重复标题
"""

DEFAULT_MODEL_NAME = ""
DEFAULT_MODEL_CHARACTER = "你是一个友好、耐心、实用的中文 AI 助手，服务于不熟悉 AI 和技术的普通用户"
CHAT_DEFAULT_FIELD = { # 一个会话被新增时应该取其中默认值的kv对
    "dirty": False,
    "prompt": "" # 为了设置提示次输入框而存在的session_state kv对
}
NOTWIDGET_FIELDS = { # 应该保存到文件中的字段的白名单(非控件字段)
    "session_id",
    "created_at",
    "updated_at",
    "name",
    "character",
    "selected_model",
    "messages",
    "title",
    "next_image_seq",
    "images",
}
DATA_DIR = Path(__file__).parent / "data"
# 创建大模型客户端
CHAT_CLIENT = openai.OpenAI(base_url=BASE_URL) # NOTE - 环境变量里设置了openai_api_key
MAX_IMAGES_PER_REQUEST = 4 # 单轮请求最多附带4张原图
MAX_IMAGES_PER_SESSION = 20 #单会话最多20张图(主要是性能考量)
MAX_API_IMAGE_PIXELS = 1_000_000 # 发送给模型的单张图片最大像素数
IMAGE_FORMAT_TO_MIMEANDSUFFIX = {
    "PNG": ["image/png", ".png"],
    "JPEG": ["image/jpeg", ".jpg"],
    "WEBP": ["image/webp", ".webp"]
}

def apply_agent_meta(agent_meta: object) -> None:
    """
    校验并保存模型返回的 AGENT_META
    只接受本轮新上传图片的记忆卡，模型不能修改历史图片记忆(忽略)

    Args:
        agent_meta (object): 模型生成的回复元数据，object类型表示接受任意对象，可以为模型生成的元数据格式定义一个专属类，这里没定义
    """
    if not isinstance(agent_meta, dict):
        st.session_state["agent_meta_error"] = "模型返回的元数据不是 JSON 对象"
        return

    title = agent_meta.get("title")
    # 如果当前会话没有标题才保存标题，避免模型误生成标题
    if st.session_state["title"] is None:
        if isinstance(title, str):
            st.session_state["title"] = title.replace("\n", " ").strip()[:12] or "新对话"
        else:
            st.session_state["title"] = "新对话"

    image_metas = agent_meta.get("images", [])
    if not isinstance(image_metas, list):
        st.session_state["agent_meta_error"] = "模型返回的图片记忆格式不正确"
        return

    latest_user_message = st.session_state["messages"][-1]
    uploaded_image_ids = latest_user_message["uploaded_image_ids"]

    # 某一条image元数据不是dict时直接忽略
    for image_meta in image_metas:
        if not isinstance(image_meta, dict):
            continue

        image_id = image_meta.get("image_id")
        summary = image_meta.get("summary")
        ocr_text = image_meta.get("ocr_text", "")

        # 校验是否有不是用户上传的图片id，摘要和ocr文字是否为str类型
        if (
            not isinstance(image_id, str)
            or image_id not in uploaded_image_ids
            or not isinstance(summary, str)
            or not summary.strip()
            or not isinstance(ocr_text, str)
        ):
            continue

        st.session_state["images"][image_id]["summary"] = summary.strip()[:30] # 只保存前30个字，后面的截掉
        st.session_state["images"][image_id]["ocr_text"] = ocr_text.strip()


def render_reasoning_chunk(reasoning_content, reasoning_placeholder):
    """
    累加并实时渲染模型生成的一个思考片段
    """
    st.session_state["current_reasoning"] += reasoning_content
    reasoning_placeholder.markdown(st.session_state["current_reasoning"])


def extract_content_stream(stream, reasoning_placeholder):
    """
    从模型原始流中提取思考内容和正式回复内容

    Args:
        stream (流): 模型返回的原始流
        reasoning_placeholder (st.empty()占位对象): 用来渲染模型思考过程的输出位置
    """
    for chunk in stream:
        delta = chunk.choices[0].delta
        # 拆出推理思考字段
        reasoning_content = getattr(delta, "reasoning", None) # 在正文输出阶段(非思考阶段), 返回的请求体中会没有reasoning字段，所以这里应该使用getattr()来获取这个字段
        if isinstance(reasoning_content, str) and reasoning_content:
            render_reasoning_chunk(reasoning_content, reasoning_placeholder)

        # 拆出正文字段
        content = delta.content
        if isinstance(content, str) and content:
            yield content

        if chunk.choices[0].finish_reason == "length": # 每条响应都检查是否出现因为上下文/max_tokens限制导致模型输出中断(据观察每条响应都有finish_reason这个字段，所以这里直接用.访问)
            st.session_state["response_truncated"] = True


def parse_agent_response_stream(content_stream):
    """
    解析模型的流式回复，隐藏 AGENT_META，只将正式回复交给聊天界面

    Args:
        content_stream (流): 只包含模型正式回复文本的流
    """
    meta_finished = False # 是否已经解析到完整元数据
    buffer = "" # 从stream已经接收到的字符集合
    st.session_state.pop("agent_meta_error", None)

    for content in content_stream:
        # 没有检查content非空，因为extract_content_stream已经检查过了
        # 解析完元数据之后就是正式回复，直接返回正式回复
        if meta_finished:
            yield content
            continue

        buffer += content # 累加当前chunk到buffer

        # 开始标记可能被拆到多个 chunk，确认不是其前缀后才能把内容当作正文输出
        if not buffer.startswith(AGENT_META_OPEN): # 如果当前buffer开头没有完整AGENT_META_OPEN
            if AGENT_META_OPEN.startswith(buffer): # 模型可能吧AGENT_META_OPEN分多个chunk返回，上面一个if进来之后可能为"[[AGEN"，只有部分AGENT_META_OPEN开头，这个if就是排除这个场景
                continue
            # 走到这里说明模型没有按照规定返回元数据，直接把当前buffer当作正式回复返回
            meta_finished = True
            st.session_state["agent_meta_error"] = "模型未按约定返回 AGENT_META"
            yield buffer
            buffer = ""
            continue

        # 找到完整开头之后寻找AGENT_META_CLOSE结尾，如果没找到就continue继续寻找
        meta_end = buffer.find(AGENT_META_CLOSE, len(AGENT_META_OPEN))
        if meta_end == -1:
            continue

        # 解析出元数据与夹带的正式回复(AGENT_META_CLOSE和部分正式回复放在同一个chunk里返回)
        meta_json = buffer[len(AGENT_META_OPEN):meta_end].strip()
        remain_body = buffer[meta_end + len(AGENT_META_CLOSE):]
        meta_finished = True

        # 下面逻辑只有刚刚解析到完整AGENT_META_CLOSE之后才会执行，解析完之后就不会再执行了
        try:
            apply_agent_meta(js.loads(meta_json))
        except js.JSONDecodeError:
            st.session_state["agent_meta_error"] = "模型返回的 AGENT_META 不是合法 JSON"

        if remain_body:
            yield remain_body

    if not meta_finished:
        st.session_state["agent_meta_error"] = "模型未返回完整的 AGENT_META"
    
            

def create_new_dialogue():
    """
    只做：新建一个默认会话，重置所有session_state状态
    """
    # 重置会话状态
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state["name"] = DEFAULT_MODEL_NAME
    st.session_state["character"] = DEFAULT_MODEL_CHARACTER
    st.session_state["messages"] = [{"role": "system", "content": SYSTEM_PROMPT_TEMPLATE % (DEFAULT_MODEL_NAME, DEFAULT_MODEL_CHARACTER)}]
    st.session_state["session_id"] = uuid.uuid4().hex
    st.session_state["created_at"] = datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%dT%H:%M:%S")
    st.session_state["updated_at"] = ""
    st.session_state["current_history_file"] = None # 当前会话对应的json文件
    st.session_state["dirty"] = False
    st.session_state["title"] = None
    st.session_state["selected_model"] = "qwen3-vl:8b"
    st.session_state["images"] = {}
    st.session_state["selected_history_image_ids"] = []
    st.session_state["next_image_seq"] = 1


def save_dialogue():
    """
    _summary_: 保存当前会话到文件
    """
    data = {
        k: v
        for k, v in st.session_state.items()
        if k in NOTWIDGET_FIELDS
    }

    #如果data目录不存在则创建目录
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(mode=0o744)
    
    # 打开文件并保存 json 到文件中
    with open(file = DATA_DIR / f"{st.session_state["session_id"]}_{st.session_state["created_at"]}.json", mode='w', encoding="utf-8") as f:
        js.dump(obj=data, fp=f, ensure_ascii=False, indent=4)

    st.session_state["dirty"] = False
    

def on_create_new_clicked():
    """
    处理用户点击新建会话之后的处理
    """
    if st.session_state["dirty"]: # 如果当前会话脏了，才保存当前会话，否则不保存
        save_dialogue()
    if st.session_state["updated_at"]: # 如果当前会话没有updated_at信息，说明已经是最新会话，再点击新建会话就没用
        create_new_dialogue()

def set_sysprompt():
    """
    用户摁下侧边栏ai特点设置确认按钮之后设置系统提示词
    """
    st.session_state["messages"][0]["content"] = SYSTEM_PROMPT_TEMPLATE % (st.session_state["name"], st.session_state["character"])
    st.session_state["dirty"] = True # 保存会话提示词之后也算会话脏了，需要保存会话
    st.session_state["sysprompt_saved"] = True

def fill_prompt():
    """
    用户摁下提示次按钮之后填充样例提示词
    """
    st.session_state["prompt"] = SAMPLE_CHAT_PROMPT

def parse_filename(path: str):
    """
    解析会话信息json文件名

    Args:
        path (str) | None: listdir返回的文件名(带后缀)，如果混了非json文件，返回None

    Returns:
        tuple[str, str, str]: (uuid, created_at)
    """
    if not path.endswith(".json"):
        return None
    uuid, created_at = path.split('_', maxsplit=1) # 根据_分割出三部分
    created_at = created_at[0:-5:] # 去掉json后缀
    return (uuid, created_at)


def switch_dialogue(file: str):
    """
    切换会话：(1).加载会话文件   (2).更新session_state  (3).清除旧的标题生成任务

    Args:
        file (str): 会话文件
    """
    # 打开文件
    with open(DATA_DIR / file) as f:
        fileds_to_update = js.load(f)
        st.session_state.update(fileds_to_update) # 转换为dict后用来更新st.session_state
    # 重置st.session_state中应取默认值的字段
    st.session_state.update(CHAT_DEFAULT_FIELD)
    st.session_state["current_history_file"] = file
    st.session_state["selected_history_image_ids"] = [] # 保留清空旧会话的用户选择历史图片列表，防止未成功删除的情况(模型还没生成回答用户旧切换会话)

def delete_dialogue(file: str):
    remove(DATA_DIR / file)
    if st.session_state["session_id"] == parse_filename(file)[0]: # type: ignore
        create_new_dialogue()

def load_session_names():
    """
    加载会话名称，用来渲染并创建历史会话按钮实例
    """
    if not DATA_DIR.exists():
        return

    for file in listdir(str(DATA_DIR)):
        # 解析文件名
        file_info = parse_filename(file)
        if not file_info: # None的情况为"非json文件"
            continue
        # 渲染按钮
        col1, col2 = st.columns([4, 1])
        with st.sidebar:
            with open(DATA_DIR/file) as f:
                title = js.load(f).get("title")
            col1.button(
                label=title if title else file_info[1],
                key=f"switch_{file_info[0]}",
                on_click=switch_dialogue,
                args=(file,),
                width="stretch",
                type="primary"
            )
            col2.button(
                label="",
                icon=DELETE_CHAT_ICON,
                key=f"delete_{file_info[0]}",
                on_click=delete_dialogue,
                args=(file,),
                width="stretch",
                type="primary"
            )

@st.fragment(run_every=2)
def schedule_task():
    """
    1.监视data目录下的json文件是否发生变更，如果变更需要重新rerun整个界面显示这些变化
    2.检查标题生成任务是否完成
    """

    # 在session_state中设置专门的 k,v对 来记录当前会话记忆中的data目录情况(使用set集合来记录最合适，可以直接用set进行比较)
    st.session_state.setdefault("known_history_files", set())

    # 获取当前data目录下的文件集合
    current_jsonfiles = {
        file
        for file in DATA_DIR.glob("*.json")
        if file.is_file()
    }

    # 比较session_state和实际情况current_jsonfiles
    if st.session_state["known_history_files"] != current_jsonfiles:
        # 下面是发历史快照和目前文件实际详情不一致，先更新session_state["known_history_files"]快照
        st.session_state["known_history_files"] = current_jsonfiles

        # 如果当前会话不是新会话，并且会话文件已经被删掉 -> 新建会话并rerun界面
        if st.session_state["current_history_file"] and not (DATA_DIR/st.session_state["current_history_file"]).exists(): # 如果当前会话不是刚刚新增的并且会话json文件不存在
            create_new_dialogue() # 新建会话
        
        st.rerun() # 重跑脚本，渲染对话与历史会话按钮
        

def handle_picures(upload_files: list[sttype.UploadedFile]) -> list[str]:
    """
    处理用户提交的图片(包含本轮图片和历史图片)

    Args:
        upload_files (UploadedFile): st.chat_input返回的files列表

    Returns:
        list[str]: 该函数对于每个被保存的图片生成的id
    """
    # 校验用户上传图片数量是否超限(单轮对话+会话总数量)
    existing_image_count = len(st.session_state["images"])
    uploaded_file_count = len(upload_files)
    history_file_count = len(st.session_state["selected_history_image_ids"]) #NOTE - 这句默认前面已经准备好了selected_history_image_ids，是个不好的代码习惯
    if uploaded_file_count + history_file_count > MAX_IMAGES_PER_REQUEST:
        raise ValueError(f"单次最多上传 {MAX_IMAGES_PER_REQUEST} 张图片，本轮携带了 {uploaded_file_count} 张图片")
    if existing_image_count + uploaded_file_count > MAX_IMAGES_PER_SESSION:
        raise ValueError(f"单个会话最多上传 {MAX_IMAGES_PER_SESSION} 张图片，当前会话图片已达 {existing_image_count + uploaded_file_count} 张图片")

    # 不采取：边一张一张校验同时保存到session_state，采取：全部校验完成之后一起保存到session_state；避免部分失败的情况
    checked_images = [] # 已校验图片
    for image in upload_files:
        image_name = image.name
        try:
            with Image.open(image) as image_file:
                image_format = image_file.format
                image_file.verify() # 校验图片完整性
        except (UnidentifiedImageError, OSError): # 校验是否为图片
            raise ValueError(f"图片 {image_name} 无法正常读取")
        if image_format not in IMAGE_FORMAT_TO_MIMEANDSUFFIX: # 校验图片格式是否受支持
            raise ValueError(f"图片 {image_name} 格式为 {image_format}，不支持")
        checked_images.append({
            "name": image_name,
            "raw_bytes": image.getvalue(),
            "mime_type": IMAGE_FORMAT_TO_MIMEANDSUFFIX[image_format][0],
            "suffix": IMAGE_FORMAT_TO_MIMEANDSUFFIX[image_format][1]
        })

    # 创建图片保存路径
    attachments_dir: Path = DATA_DIR / "attachments" / st.session_state["session_id"] # 图片保存绝对路径
    attachments_dir.mkdir(mode=0o744, exist_ok=True)

    image_ids = [] # 生成的图片id列表

    for checked_image in checked_images:
        image_id = f"img_{st.session_state['next_image_seq']:03d}"
        saved_path = attachments_dir / f"{image_id}{checked_image["suffix"]}" # 需要保存的图片绝对路径
        saved_path.write_bytes(checked_image["raw_bytes"]) # 保存图片到saved_path
        st.session_state["images"][image_id] = {
            "name": checked_image["name"],
            "path": str(saved_path.relative_to(DATA_DIR)),
            "mime_type": checked_image["mime_type"],
            "summary": "摘要尚未生成",
            "ocr_text": ""
        }
        image_ids.append(image_id)
        st.session_state["next_image_seq"] += 1

    return image_ids


def reject_user_submission(prompt: str, error_info: str) -> None:
    """
    中止本轮提交，保留用户输入的文字并在下一次重跑后显示错误
    """
    st.session_state["last_rounderror"] = {
        "prompt": prompt,
        "error_info": error_info
    }
    st.rerun()


def check_last_image_error():
    error = st.session_state.pop("last_rounderror", None)
    if not error:
        return
    st.session_state["prompt"] = error["prompt"]
    st.error(body=error["error_info"], icon='🚨')

def render_image_checkboxs() -> None:
    """
    渲染侧边栏复选框
    根据session_satte中保存的本轮涉及图片信息，为每幅图片渲染复选框(复选框与缩略图并排放置)，供用户选择本轮交互是否携带这张图
    会修改session_state，selected_history_image_ids(list[str])包含被选择的图片的id
    """
    selected_ids: list[str] = []

    with st.sidebar.expander("历史图片"):
        if not st.session_state["images"]:
            st.caption("当前会话还没有历史图片")

        for image_id, info in st.session_state["images"].items():
            image_col, selected_col = st.columns([1, 2])
            with image_col:
                st.image(DATA_DIR/info["path"], width=100) # 该图片大小主要受侧边栏宽度影响
            with selected_col:
                selected = st.checkbox(
                    label=image_id,
                    key=f"history_image_{st.session_state['session_id']}_{image_id}"
                )
                st.caption(info["summary"])
            if selected:
                selected_ids.append(image_id)

    # 直接赋值到session_state["selected_history_image_ids"]而不是append，追加不符合逻辑！！
    st.session_state["selected_history_image_ids"] = selected_ids

def build_image_catalogue() -> str:
    """
    构造所有已存在历史图片的摘要目录

    Returns:
        str: 历史图片摘要目录，格式：
        - image_001：<summary>；OCR：{ocr_text}
        - image_002：<summary>；OCR：{ocr_text}
        ...
    """
    images = st.session_state.get("images", None)
    if not images:
        return "无"
    lines: list[str] = []
    for id, info in images.items():
        summary = info["summary"].strip()
        ocr_text = info["ocr_text"].strip()
        lines.append(f"- {id}：{summary}；OCR：{ocr_text}")
    return '\n'.join(lines)


def build_image_context_prompt() -> str:
    """
    生成每轮动态图片提示词

    历史图片目录会每轮发送；
    勾选历史图和新上传图在请求体中会额外附带原图

    Returns:
        str: 动态图片提示词，格式见IMAGE_PROMPT
    """
    catalogue = build_image_catalogue()

    latest_user_message = st.session_state["messages"][-1] # 从最新用户消息生成提示词
    # 用户选择的历史图片
    selected_history_image_ids: list[str] = latest_user_message["user_selected_image_ids"]
    selected_text: str = ("、".join(selected_history_image_ids) if selected_history_image_ids else "无")
    # 用户这轮新上传的图片
    new_image_ids: list[str] = latest_user_message["uploaded_image_ids"]
    new_text = "、".join(new_image_ids) if new_image_ids else "无"
    return IMAGE_PROMPT % (catalogue, selected_text, new_text)


def image_id_to_api_content(image_id: str) -> dict:
    """
    读取本地原图(单张)，临时转换为 OpenAI 兼容的 image_url
    内部会将超过像素限制的图片压缩到限制以下
    
    Args:
        image_id (str): 图片id

    Returns:
        dict: 内容块字典
        格式为:
        {
            "type": "image_url",
            "image_url": "data:image/png;base64,<图片base64字符串>"
        }
    """
    info = st.session_state["images"][image_id]
    path: Path = DATA_DIR / info["path"]
    if not path.is_file():
        raise FileNotFoundError(f"找不到图片: {path}")
    with Image.open(path) as image_file:
        # 获取原图片 长，宽，像素
        width, height = image_file.size
        image_pixels = width * height

        # 如果该图片像素大于单图片最大像素限制，则需要进行压缩
        if image_pixels <= MAX_API_IMAGE_PIXELS:
            image_bytes = path.read_bytes()
        else:
            # 根据像素进行压缩，目标为将像素压缩到最大像素限制以下，这里计算长宽压缩比例
            scale = math.sqrt(MAX_API_IMAGE_PIXELS / image_pixels)
            resized_size = (
                math.floor(width * scale),
                math.floor(height * scale)
            )
            # 根据前面计算出的长宽进行压缩，采取保留最完成的LANCZOS重采样算法
            resized_image = image_file.resize(
                size=resized_size,
                resample=Image.Resampling.LANCZOS
            )
            # 防止用户选择jpeg保存图片但是图片中包含alpha透明通，解决方案是转换为不含透明通道模式
            if info["mime_type"] == "image/jpeg" and resized_image.mode in {"RGBA", "LA"}:
                resized_image = resized_image.convert("RGB")
            # 将Image对象转换为对应格式图片的二进制字节
            image_buffer = BytesIO()
            resized_image.save(image_buffer, format=image_file.format)
            image_bytes = image_buffer.getvalue()
            # 添加记录: 哪些图片压缩像素
            st.session_state["compressed_images"].append({
                "image_id": image_id,
                "name": info.get("name", image_id),
                "original_size": (width, height),
                "resized_size": resized_size
            })
    # 进行base64编码
    encoded_str = base64.b64encode(image_bytes).decode("ascii")
    return {
        "type": "image_url",
        "image_url": f"data:{info['mime_type']};base64,{encoded_str}"
    }


def build_api_messages() -> list[dict]:
    """
    将内部会话消息st.session_state["messages"]重新组装为 OpenAI api 格式的消息
    主要工作:
        (1).生成图片相关提示词
        (2).用户历史消息重新生成提示词(主要是st.session_state["messages"]列表项中两个图片相关字段)
        (3).用户最新提示词涉及的图片组装content部分: 包含用户提示词，每张图片对应的图片说明(每张图片前面添加text类型说明这张图片对应的image_id)与每张图片对应的base64

    Returns:
        list[dict]: _description_
    """
    raw_messages = st.session_state["messages"]
    #组装第一条系统提示词
    sys_prompt_content = raw_messages[0]["content"] # 字符串不可变对象，不用copy()
    if sum(message["role"] == "user" for message in raw_messages) == 1: # 如果原始消息列表中只有一条用户消息，则为第一条用户消息
        sys_prompt_content += TITLE_GENERATE_PROMPT

    # 组装图片相关提示词
    api_messages: list[dict] = [ # 下面为前两条系统提示词: (1)ai名称，性格 (2)图片相关说明
        {
            "role": "system",
            "content": sys_prompt_content,
        },
        {
            "role": "system",
            "content": build_image_context_prompt()
        },
    ]

    st.session_state["compressed_images"] = [] # 在这轮图片请求构造开始前先将compressed_images记录列表遗留的上一轮结果清空

    now = 2 # raw_messages[1] 对应第一条用户消息，其位置号为 2
    # 组装用户历史提示词/模型回复
    for raw_message in raw_messages[1::]:
        is_latest_message = now == len(raw_messages)
        now += 1
        role = raw_message["role"]
        content = raw_message["content"]

        # 模型消息直接放进api_messages然后进入下一轮消息组装
        if role == "assistant":
            api_messages.append({
                "role": "assistant",
                "content": content
            })
            continue

        # 用户消息(这里区分历史图片与本轮上传图片，方便后面对每张图片说明是历史图片不生成记忆卡 还是 新图片需要生成记忆卡)
        uploaded_image_ids = raw_message["uploaded_image_ids"]
        selected_image_ids = raw_message["user_selected_image_ids"]
        all_images = list(dict.fromkeys(uploaded_image_ids + selected_image_ids))

        # 历史消息，或最新但未附带原图的消息，只发送文字和关联图片 ID
        if not is_latest_message or not all_images:
            if all_images:
                content += "\n[本轮总共涉及需要详细查看的图片: " + '、'.join(all_images) + ']'
            api_messages.append({
                "role": "user",
                "content": content
            })
            continue
        # 非最新用户消息/不带图的最新消息到此处理完毕，下面为带图最新消息的逻辑

        # 最新消息，组装content部分图片说明与base64字符串
        content_parts: list[dict] = [{ # 对于图片请求需要的
            "type": "text",
            "text": content or "请描述这些图片"
        }]


        # 下面在append(image_id_to_api_content())之前携带每张原图前紧挨对应的 image_id 说明，模型才能建立图片与内部编号的映射

        # 历史图片
        for image_id in selected_image_ids:
            content_parts.append({
                "type": "text",
                "text": (
                    f"【以下附带原图对应历史图片 {image_id}】"
                    "仅用于本轮精确回答，不要重新生成图片记忆卡"
                )
            })
            content_parts.append(image_id_to_api_content(image_id))
        # 本轮上传图片
        for image_id in uploaded_image_ids:
            content_parts.append({
                "type": "text",
                "text": (
                    f"【以下附带原图对应本轮新上传图片 {image_id}】"
                    "请为该图片生成图片记忆卡"
                )
            })
            content_parts.append(image_id_to_api_content(image_id)) # 添加原图对应的 openai api content 块
        api_messages.append({ # 向api_messages添加本轮请求对应的请求体
            "role": "user",
            "content": content_parts
        })
    return api_messages


def clear_history_image_selection_if_needed() -> None:
    """
    在复选框创建前，清理上一轮发送成功后遗留的勾选状态
    """
    should_clear = st.session_state.pop("should_clear_history_image_selection", False) # 这句有三个含有: (1)取出清除标记 (2)如果有清除标记, pop()会删掉清楚标记 (3)如果没有清楚标记，默认不清除

    if not should_clear:
        return

    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith("history_image_"):
            st.session_state[key] = False # 设置全部复选框为取消状态

    st.session_state["selected_history_image_ids"] = []


def main():
    # 界面基本设置
    st.logo(LOGO, icon_image=LOGO)
    st.set_page_config(page_title="自制ai聊天网页",
                       page_icon="💬",
                       layout="wide")
    st.title("AI聊天伴侣")

    # 第一次进入网页/刷新时创建一个新会话
    if "session_id" not in st.session_state:
        create_new_dialogue()

    st.caption(f"当前会话：{st.session_state['title'] or '新对话'}")

    # 必须在 st.checkbox 创建前清空，才能安全重置勾选状态
    clear_history_image_selection_if_needed()

    # 侧边栏
    with st.sidebar:
        st.subheader("会话管理")
        st.button("新建会话", icon=NEW_CHAT_ICON, width="stretch", on_click=on_create_new_clicked)
        load_session_names()
        st.divider()

        st.subheader("伴侣信息")
        st.text_input(label="伴侣姓名", key="name")
        st.text_area(label="伴侣性格", key="character")
        st.button("确定", on_click=set_sysprompt)
        if st.session_state.pop("sysprompt_saved", False):
            st.toast("成功应用伴侣信息", icon='😀')
        st.divider()

        render_image_checkboxs() # 渲染历史图片复选框


    # 渲染历史聊天记录
    for message in st.session_state["messages"][1::]:
        with st.chat_message(name=message["role"]):
            # 先逐张渲染用户上传的图片
            if message["role"] == "user":
                for image_id in message["uploaded_image_ids"]:
                    image_info = st.session_state["images"][image_id]
                    image_path = DATA_DIR / image_info["path"]
                    if image_path.is_file():
                        st.image(image_path, width=250)
                    else:
                        st.warning(f"图片 {image_id} 的原文件不存在")
            # 再渲染用户提示词
            st.markdown(message["content"])

    # 渲染在页面最底部
    with st.bottom:  # noqa: SIM117
        with st.container(border=True):
            check_last_image_error() # 检查上一轮是否有图片异常错误(内部有填写st.session_state["prompt"]，必须放在chat_input渲染完成之前)
            st.button(label="提示词例子", on_click=fill_prompt)
            # 聊天框
            user_input = st.chat_input(
                placeholder="输入消息或者直接粘贴图片...",
                key="prompt",
                accept_file="multiple",
                file_type=["jpg", "jpeg", "png"]
            )

    schedule_task() # 必须放在if not prompt前面调用，因为如果当前会话用户还从未填写过提示词，只是把这个会话加载出来，这个函数就不会被调用，也就不会注册这个fragment函数

    # 用户只勾选历史图片时，构造空文字和空上传文件的默认输入，后续仍可走统一提交流程
    if not user_input:
        if not st.session_state["selected_history_image_ids"]:
            return
        user_input = {
            "text": "请描述这些图片",
            "files": []
        }

    # 处理本轮请求相关图片
    image_ids: list[str] = []
    try:
        image_ids = handle_picures(user_input["files"]) # 内部保存并校验用户上传的图片，验失败会抛出异常，进入异常捕获分支直接rerun()，不会进入下面渲染用户本轮消息的脚本
    except ValueError as e:
        reject_user_submission(user_input["text"], str(e))

    # 渲染用户本轮消息用户文字和图片放在同一个聊天消息区域内渲染
    with st.chat_message(name="user"):
        if user_input["text"]:
            st.write(user_input["text"])
        # 渲染用户新上传图片
        for image in user_input["files"]:
            st.image(image, width=250)
        # 渲染用户历史图片
        for image_id in st.session_state["selected_history_image_ids"]:
            image_info = st.session_state["images"][image_id]
            image_path = DATA_DIR / image_info["path"]
            if image_path.is_file():
                st.image(image_path, width=250)
            else:
                st.warning(f"图片 {image_id} 的原文件不存在")

    
    st.session_state["messages"].append({
        "role": "user", 
        "content": user_input["text"],
        "uploaded_image_ids": image_ids,
        "user_selected_image_ids": st.session_state["selected_history_image_ids"].copy() #这里使用拷贝而不是直接赋值是因为list是可变对象，直接赋值之后修改st.session_state["selected_history_image_ids"](每轮交互结束后清空)相当于一起修改了这条消息的user_selected_image_ids，不符合逻辑
    })

    # 调用大模型生成回答
    #TEST
    api_messages = build_api_messages()
    request_debug_file = Path(__file__).parent / "debug_api_messages.json"
    request_debug_file.write_text(
        js.dumps(api_messages, ensure_ascii=False, indent=4),
        encoding="utf-8"
    )
    #TEST

    # 显示给用户哪些图片被调整了大小
    for image_info in st.session_state.get("compressed_images", []): # 这里使用get获取，当不存在(首次发送消息时不携带图片)时返回空列表，也是一个不好的代码习惯
        original_width, original_height = image_info["original_size"]
        resized_width, resized_height = image_info["resized_size"]
        st.warning(
            f"图片 {image_info['name']}({image_info["image_id"]}) 发送时已从 {original_width} × {original_height} "
            f"压缩为 {resized_width} × {resized_height}"
        )
    
    st.session_state["current_reasoning"] = "" # 接收模型生成的回复钱先清空上一轮的思考内容
    stream_response = CHAT_CLIENT.chat.completions.create(
        model=st.session_state["selected_model"],
        messages=build_api_messages(), # type: ignore
        stream=True,
        timeout=180.0
    ) # type: ignore

    # 流式输出大模型的回复
    with st.chat_message(name="assistant"):
        with st.expander("思考过程"): # 下拉框: 用户可选是否查看思考过程
            reasoning_placeholder = st.empty()
        # 获取两个流生成器对象(还未启动流): 模型思考过程和正文回复
        content_stream = extract_content_stream(stream_response, reasoning_placeholder)
        stream_gen = parse_agent_response_stream(content_stream)
        # 下面这句才真正驱动流进行输出(该函数看起来只渲染正式回复，但是内部的流逻辑会更新思考过程占位块)
        full_reply = st.write_stream(stream=stream_gen)

    agent_meta_error = st.session_state.pop("agent_meta_error", None)
    if agent_meta_error:
        st.warning(f"{agent_meta_error}，本轮图片记忆可能未保存")

    if st.session_state.pop("response_truncated", False):
        st.warning("本轮回复可能因上下文空间不足而被截断")

    # 将ai生成的回复也添加到messages历史消息列表中
    st.session_state["messages"].append({
        "role": "assistant",
        "content": full_reply
    })
    st.session_state["updated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["dirty"] = True

    save_dialogue()
    st.session_state["should_clear_history_image_selection"] = True
    st.rerun() # 不会不停重新跑，因为前面有if not user_input: return这句入贵没有提示词会提前退出



if __name__ == "__main__":
    main()

#FIXME - BUG: 如果模型上一条对话还为发送完就发送下一条对话，会吧上一条对话给“遮掉”
