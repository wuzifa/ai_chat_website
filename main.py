import streamlit as st

logo_path = "./chat_robot.png"
sample_chat_prompt = "你能帮我做什么？"


# 界面基本设置
st.logo("./chat_robot.png", icon_image="./chat_robot.png")
st.set_page_config(page_title="自制ai聊天网页",
                   page_icon="💬",
                   layout="wide")
st.title("AI聊天伴侣")
#TODO - 添加当前会话名称

# 预天聊天框按钮
if st.button(label="提示词例子"):
    st.session_state["prompt"] = sample_chat_prompt

# 聊天框
prompt = st.chat_input(placeholder="随便说点什么...", key="prompt")
print(f"用户提示词：{prompt}")

# 调用大模型生成回答