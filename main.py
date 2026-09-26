import openai
import streamlit as st

LOG_PATH = "./chat_robot.png"
SAMPLE_CHAT_PROMPT = "你能帮我做什么？"
BASE_URL = "http://192.168.1.23:11434/v1"



# 界面基本设置
st.logo("./chat_robot.png", icon_image="./chat_robot.png")
st.set_page_config(page_title="自制ai聊天网页",
                   page_icon="💬",
                   layout="wide")
st.title("AI聊天伴侣")
#TODO - 添加当前会话名称

# 创建大模型客户端
client = openai.OpenAI(base_url=BASE_URL) # NOTE - 环境变量里设置了openai_api_key

# 预天聊天框按钮
if st.button(label="提示词例子"):
    st.session_state["prompt"] = SAMPLE_CHAT_PROMPT

# 聊天框
prompt = st.chat_input(placeholder="随便说点什么...", key="prompt")
print(f"用户提示词：{prompt}")



# 如果用户有实际输入
if prompt:
    # 渲染用户提示词
    st.chat_message(name="user").write(prompt)
    # 调用大模型生成回答
    stream_response = client.chat.completions.create(
        model="deepseek-r1:8b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt}
        ],
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


#FIXME - 用户界面不显示当前模型正在思考的内容，但是在apifox里面可以看到流式完整思考过程，用户体验很不好，我希望像codex app，可以显示思考过程，并且是一个下拉框，用户可以点击选择是否显示思考内容

