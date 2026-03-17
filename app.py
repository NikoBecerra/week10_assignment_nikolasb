import requests
import streamlit as st

API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = "meta-llama/Llama-3.2-1B-Instruct"

st.set_page_config(page_title="My AI Chat", layout="wide")


def load_hf_token() -> str | None:
	try:
		token = st.secrets["HF_TOKEN"]
	except Exception:
		return None

	token = str(token).strip()
	return token if token else None


def send_message(token: str, messages: list) -> str:
	headers = {"Authorization": f"Bearer {token}"}
	payload = {
		"model": MODEL,
		"messages": messages,
		"max_tokens": 512,
	}

	response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
	response.raise_for_status()
	data = response.json()
	return data["choices"][0]["message"]["content"]


# Initialize session state for conversation history
if "messages" not in st.session_state:
	st.session_state.messages = []

st.title("My AI Chat")
st.write("Part B: Multi-Turn Conversation UI")

hf_token = load_hf_token()
if not hf_token:
	st.error(
		"Missing or empty HF token. Add HF_TOKEN to .streamlit/secrets.toml (or Streamlit Cloud Secrets)."
	)
	st.stop()

# Display conversation history
for message in st.session_state.messages:
	with st.chat_message(message["role"]):
		st.write(message["content"])

# Fixed input bar at the bottom
user_input = st.chat_input("Type your message here...")

if user_input:
	# Add user message to history
	st.session_state.messages.append({"role": "user", "content": user_input})
	
	# Send message to API with full history
	with st.spinner("Sending message..."):
		try:
			assistant_response = send_message(hf_token, st.session_state.messages)
			
			# Add assistant response to history
			st.session_state.messages.append({"role": "assistant", "content": assistant_response})
			
		except requests.HTTPError as http_err:
			status = http_err.response.status_code if http_err.response is not None else "unknown"
			if status == 401:
				st.error("Invalid token (401). Check HF_TOKEN in secrets.")
			elif status == 429:
				st.error("Rate limit hit (429). Please retry shortly.")
			else:
				st.error(f"API request failed (status {status}).")
			detail = http_err.response.text if http_err.response is not None else str(http_err)
			st.caption(detail)
		except requests.RequestException as req_err:
			st.error("Network failure while contacting Hugging Face API.")
			st.caption(str(req_err))
		except (KeyError, IndexError, TypeError, ValueError) as parse_err:
			st.error("Unexpected response format from Hugging Face API.")
			st.caption(str(parse_err))
	
	st.rerun()

