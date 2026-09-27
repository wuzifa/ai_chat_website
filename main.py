import json as js
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import openai
import streamlit as st

LOG_PATH = "./chat_robot.png"
SAMPLE_CHAT_PROMPT = "你能帮我做什么？"
BASE_URL = "http://192.168.1.23:11434/v1"
SYSTEM_PROMPT_TEMPLATE = """
    用户要求：
    你的名字：%s
    你的性格：%s
"""
DEFAULT_MODEL_NAME = ""
DEFAULT_MODEL_CHARACTER = "你是一个友好、耐心、实用的中文 AI 助手，服务于不熟悉 AI 和技术的普通用户"
selected_model = "qwen3-vl:8b"
dialogue_id = datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime("%y-%m-%d_%H-%M-%S")

def save_dialogue():
    """
    _summary_: 保存当前会话到文件
    """
    # 组装要保存到文件中的信息
    data = {
        "name": st.session_state["name"],
        "character": st.session_state["character"],
        "messages": st.session_state["messages"][1::] #只保留对话信息，不保存系统提示词
    }
    #如果data目录不存在则创建目录
    data_dir = Path(__file__).parent / "data"
    if not data_dir.exists():
        data_dir.mkdir(mode=0o744)
    # 打开文件并保存 json 到文件中
    with open(file=f"./data/{dialogue_id}.json", mode='w', encoding="utf-8") as f:
        js.dump(obj=data, fp=f, ensure_ascii=False, indent=4)
    
def reset_dialogue():
    global messages
    st.session_state["name"] = DEFAULT_MODEL_NAME
    st.session_state["character"] = DEFAULT_MODEL_CHARACTER
    messages = [{"role": "system", "content": SYSTEM_PROMPT_TEMPLATE % (DEFAULT_MODEL_NAME, DEFAULT_MODEL_CHARACTER)}]

# 初始化会话基本属性
# 默认系统提示词
messages = st.session_state.setdefault(
    "messages", 
    [{
        "role": "system",
        "content": SYSTEM_PROMPT_TEMPLATE % (st.session_state.setdefault("name", DEFAULT_MODEL_NAME), st.session_state.setdefault("character", DEFAULT_MODEL_CHARACTER))
    }],
)
#当前会话是否有必要保存
should_save = st.session_state.setdefault("should_save", False) #FIXME - 添加维护该变量的部分

# 界面基本设置
st.logo("./chat_robot.png", icon_image="./chat_robot.png")
st.set_page_config(page_title="自制ai聊天网页",
                   page_icon="💬",
                   layout="wide")
st.title("AI聊天伴侣")
#TODO - 添加当前会话名称

# 创建大模型客户端
client = openai.OpenAI(base_url=BASE_URL) # NOTE - 环境变量里设置了openai_api_key


# 侧边栏
with st.sidebar:
    st.subheader("会话管理")
    if st.button("新建会话"):
        save_dialogue()
        reset_dialogue() # 内部会清空messages列表，这里把整个侧边栏的绘制放在正文界面的渲染上面，免去写rerun()

    st.divider()

    st.subheader("伴侣信息")
    name = st.text_input(label="伴侣姓名", key="name")
    character = st.text_area(label="伴侣性格", key="character")

    if st.button("确定"):
        st.session_state["messages"][0]["content"] = SYSTEM_PROMPT_TEMPLATE % (name, character)


# 渲染历史聊天记录
for message in messages[1::]:
    with st.chat_message(name=message["role"]):
        st.markdown(message["content"])

# 渲染在页面最底部
with st.bottom:  # noqa: SIM117
    with st.container(border=True):
        # 预填聊天框按钮
        if st.button(label="提示词例子"):
            st.session_state["prompt"] = SAMPLE_CHAT_PROMPT
        # 聊天框
        prompt = st.chat_input(placeholder="随便说点什么...", key="prompt")


# 如果用户有实际输入
if prompt:
    # 渲染用户提示词
    st.chat_message(name="user").write(prompt)
    messages.append({
        "role": "user", 
        "content": prompt
    })
    # 调用大模型生成回答
    stream_response = client.chat.completions.create(
        model=selected_model,
        messages=messages,
        stream=True
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
    messages.append({
        "role": "assistant", 
        "content": full_reply
    })



#TODO - 用户界面不显示当前模型正在思考的内容，但是在apifox里面可以看到流式完整思考过程，用户体验很不好，我希望像codex app，可以显示思考过程，并且是一个下拉框，用户可以点击选择是否显示思考内容
#TODO - 点击确定按钮更改ai性格成功之后应该给个提示，否则用户不知道是否应用成功  (总是忘记设置点击确定，应该有个更能明显的提示)
#TODO - 尝试实现让ai总结对话标题(我的思路：从空对话开始用户发起对话时，发起两个请求，一个用来回复用户，另一个用来总结描述用户的问题来生成一个合适的对话标题，让回复用户的请求先运行)
#FIX - 当窗口切换到streamlit外面(例如我在查看vscode代码)，streamlit没有渲染ai生成的回复(好像streamlit停止渲染了)
# 添加会话记忆功能
# 使用st.session_state设置一个messages列表来保存历史会话，每轮对话需要像里面append对话
# 发送提示词则可以直接使用st.session_state
