import openai
import streamlit as st

LOG_PATH = "./chat_robot.png"
SAMPLE_CHAT_PROMPT = "你能帮我做什么？"
BASE_URL = "http://192.168.1.23:11434/v1"
SYSTEM_PROMPT = """
    你是一个友好、耐心、实用的中文 AI 助手，服务于不熟悉 AI 和技术的普通用户。

    回答要求：
    - 使用简单、自然的中文，避免专业词；必须使用时，顺手用一句普通话解释。
    - 先直接回答用户最关心的问题，再补充必要说明。
    - 默认简短清楚，通常不超过 3 个要点；用户说“详细一点”时再展开。
    - 不说空话，不寒暄，不重复用户的问题。
    - 提建议时给具体、容易执行的做法，不一次给太多选择。
    - 用户的问题不完整时，先按最常见的理解回答；只有确实无法回答时，才问一个最关键的问题。
    - 不确定或无法确认的信息，要直接说明，不要猜测或编造。
    - 用户上传图片时，根据图片中能看清的内容回答；看不清时，用简单的话说明原因。
    - 默认使用中文；用户明确要求时再切换语言。
"""
selected_model = "qwen3-vl:8b"

# 初始化会话基本属性
messages = st.session_state.setdefault(
    "messages", 
    [{"role": "system", "content": SYSTEM_PROMPT}],
) # 默认系统提示词

# 界面基本设置
st.logo("./chat_robot.png", icon_image="./chat_robot.png")
st.set_page_config(page_title="自制ai聊天网页",
                   page_icon="💬",
                   layout="wide")
st.title("AI聊天伴侣")
#TODO - 添加当前会话名称

# 创建大模型客户端
client = openai.OpenAI(base_url=BASE_URL) # NOTE - 环境变量里设置了openai_api_key

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
        print(f"用户提示词：{prompt}")


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

#FIXME - 用户界面不显示当前模型正在思考的内容，但是在apifox里面可以看到流式完整思考过程，用户体验很不好，我希望像codex app，可以显示思考过程，并且是一个下拉框，用户可以点击选择是否显示思考内容
# 添加会话记忆功能
# 使用st.session_state设置一个messages列表来保存历史会话，每轮对话需要像里面append对话
# 发送提示词则可以直接使用st.session_state


