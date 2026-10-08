import json as js
import uuid  # 生成会话唯一标识
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from os import listdir, remove
from pathlib import Path
from zoneinfo import ZoneInfo

import openai
import streamlit as st

LOGO = "./chat_robot.png"
NEW_CHAT_ICON = ":material/add_comment:"
DELETE_CHAT_ICON = ":material/delete:"
FILE_SAVE_ICON = ":material/file_save:"
SAMPLE_CHAT_PROMPT = "你能帮我做什么？"
BASE_URL = "http://192.168.1.23:11434/v1"
SYSTEM_PROMPT_TEMPLATE = """
    语言：使用中文回答

    用户要求：
    你的名字：%s
    你的性格：%s
"""
GENERATE_TITLE_PROMPT = """
    根据下面的用户首条消息生成聊天标题
    要求：
    1. 使用中文；
    2. 不超过 12 个字；
    3. 只返回标题，不要引号、解释、序号；
"""
DEFAULT_MODEL_NAME = ""
DEFAULT_MODEL_CHARACTER = "你是一个友好、耐心、实用的中文 AI 助手，服务于不熟悉 AI 和技术的普通用户"
CHAT_DEFAULT_FIELD = { # 一个会话被新增时应该取其中默认值的kv对
    "dirty": False,
    "prompt": "", # 为了设置提示次输入框而存在的session_state kv对
    "title": ""
}
NOTWIDGET_FIELDS = { # 应该保存到文件中的字段的白名单(非控件字段)
    "session_id",
    "created_at",
    "updated_at",
    "name",
    "character",
    "selected_model",
    "messages",
    "title"
}
DATA_DIR = Path(__file__).parent / "data"
# 创建大模型客户端
CHAT_CLIENT = openai.OpenAI(base_url=BASE_URL) # NOTE - 环境变量里设置了openai_api_key
TITLE_CLIENT = openai.OpenAI(base_url=BASE_URL)

def clear_title_task():
    """
    (1).清除会话的异步标题请求所有相关上下文，防止session_state中旧会话未完成的标题异步请求带到新会话导致标题赋给新会话
    (2).删除后schedule_task()函数中的"if future is None"语句也能阻止同一个会话的后续请求继续进入后续获取future结果的逻辑
    """
    if st.session_state.get("title_future"):
        st.session_state["title_future"].cancel()
    del st.session_state["title_future"]

def create_new_dialogue():
    """
    只做：新建一个默认会话，重置所有session_state状态
    """
    # 重置会话状态
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
    切换会话：(1).加载会话文件   (2).更新session_state

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
    
    """
    2.检查标题生成任务是否完成
    """
    future = st.session_state.get("title_future")
    if future is not None and future.done(): # 先判断None: 不是每个会话都会有标题任务: (1).用户还没发送第一条消息 (2).当前会话不是新会话 (3).标题任务已完成，并且已删除 title_future
        try:
            st.session_state["title"] = future.result()
        except Exception as e:  # noqa: BLE001
            st.session_state["title"] = "新对话"
            print(e)
        save_dialogue()
        # 删除两个key，通过上面的"if future is None"语句来阻止进入这块逻辑
        



def generate_title(first_user_message: str, model: str) -> str:
    """
    生成会话标题的函数，应被threadpoolexecutor异步调用

    Args:
        first_user_message (str): 用户首条消息
        model (str) : 选择的模型

    Returns:
        str: 模型生成的标题
    """
    response = TITLE_CLIENT.chat.completions.create(
           model=model,
           messages=[
               {"role": "system", "content": GENERATE_TITLE_PROMPT},
               {"role": "user", "content": first_user_message}
           ],
           stream=False,
           timeout=90.0
       )
    title = (response.choices[0].message.content or "新对话").strip() # 模型生成的回复
    return title.replace('\n', ' ')[:12:]

def on_newchat_clicked():
    save_dialogue() # 把st.session_state["dirty"] = False这句拆出save_dialogue是因为保存标题调用了save_dialogue()
    st.session_state["dirty"] = False


def main():
    # 界面基本设置
    st.logo(LOGO, icon_image=LOGO)
    st.set_page_config(page_title="自制ai聊天网页",
                       page_icon="💬",
                       layout="wide")
    st.title("AI聊天伴侣")
    #TODO - 添加当前会话名称

    # 初始化一个tltle创建线程
    st.session_state.setdefault(
        "title_generator",
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="title")
    )

    # 第一次进入网页/刷新时创建一个新会话
    if "session_id" not in st.session_state:
        create_new_dialogue()

    # 侧边栏
    with st.sidebar:
        st.subheader("会话管理")
        st.button("新建会话", icon=NEW_CHAT_ICON, width="stretch", on_click=on_create_new_clicked)
        st.button("保存会话", icon=FILE_SAVE_ICON, width="stretch", on_click=on_newchat_clicked)
        load_session_names()

        st.divider()

        st.subheader("伴侣信息")
        st.text_input(label="伴侣姓名", key="name")
        st.text_area(label="伴侣性格", key="character")

        st.button("确定", on_click=set_sysprompt)


    # 渲染历史聊天记录
    for message in st.session_state["messages"][1::]:
        with st.chat_message(name=message["role"]):
            st.markdown(message["content"])

    # 渲染在页面最底部
    with st.bottom:  # noqa: SIM117
        with st.container(border=True):
            # 预填聊天框按钮
            st.button(label="提示词例子", on_click=fill_prompt)
            # 聊天框
            prompt = st.chat_input(placeholder="随便说点什么...", key="prompt")

    schedule_task() # 必须放在if not prompt前面调用，因为如果当前会话用户还从未填写过提示词，只是把这个会话加载出来，这个函数就不会被调用，也就不会注册这个fragment函数

    # 如果用户没有实际输入，下面的新一轮对话就不用渲染了
    if not prompt:
        return

    # 先判断是否为第一条用户消息 #NOTE - 也可以使用updated_at来判断是否为第一条消息
    not_first = any(
        message["role"] == "user"
        for message in st.session_state["messages"]
    )

    # 渲染用户提示词
    st.chat_message(name="user").write(prompt)
    st.session_state["messages"].append({
        "role": "user", 
        "content": prompt
    })
    # 调用大模型生成回答
    stream_response = CHAT_CLIENT.chat.completions.create(
        model=st.session_state["selected_model"],
        messages=st.session_state["messages"],
        stream=True,
        timeout=60.0
    )
    # 流式输出大模型的回复
    with st.chat_message(name="assistant"):
        full_reply = st.write_stream(stream=
            (
                chunk.choices[0].delta.content or "" 
                for chunk in stream_response
            )
        )
    # 将ai生成的回复也添加到messages历史消息列表中
    st.session_state["messages"].append({
        "role": "assistant", 
        "content": full_reply
    })
    st.session_state["updated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["dirty"] = True
    #NOTE - 到这里回复请求才完毕，再发起标题生成请求，不把标题生成请求放在回复生成请求前面
    if not not_first:
        st.session_state["title_future"] = st.session_state["title_generator"].submit(generate_title, prompt, st.session_state["selected_model"])



if __name__ == "__main__":
    main()

#TODO - 用户界面不显示当前模型正在思考的内容，但是在apifox里面可以看到流式完整思考过程，用户体验很不好，我希望像codex app，可以显示思考过程，并且是一个下拉框，用户可以点击选择是否显示思考内容
#TODO - 点击确定按钮更改ai性格成功之后应该给个提示，否则用户不知道是否应用成功  (总是忘记设置点击确定，应该有个更能明显的提示)
#TODO - 尝试实现让ai总结对话标题(我的思路：从空对话开始用户发起对话时，发起两个请求，一个用来回复用户，另一个用来总结描述用户的问题来生成一个合适的对话标题，让回复用户的请求先运行)
#FIX - 当窗口切换到streamlit外面(例如我在查看vscode代码)，streamlit没有渲染ai生成的回复(好像streamlit停止渲染了)
# 添加会话记忆功能
# 使用st.session_state设置一个messages列表来保存历史会话，每轮对话需要像里面append对话
# 发送提示词则可以直接使用st.session_state

#FIXME - BUG: 如果模型上一条对话还为发送完就发送下一条对话，会吧上一条对话给“遮掉”
#FIXME - 将标题融在文件名里会出现下面情况：在标题未生成前就保存会话，save_dialogue之后出现旧的和新的两份会话文件，并且保存会话之后操作的还是旧的会话文件(st.session_state["current_history_file"]没改)，可能导致同一个会话重复保存不同名会话文件; 同时 current_history_file 也没有在保存后更新
#将文件名在对话保存的时候就固定下来，在json中保存文件名，在加载侧边栏会话时读取json文件将文件名显示在按钮中