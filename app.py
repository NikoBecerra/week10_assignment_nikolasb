import requests
import streamlit as st

API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def get_hf_token() -> str | None:
	token = st.secrets.get("HF_TOKEN")
	if not token:
		return None
	return str(token).strip()


def chat_completion(user_message: str, token: str) -> str:
	headers = {
		"Authorization": f"Bearer {token}",
	}
	payload = {
		"model": MODEL,
		"messages": [{"role": "user", "content": user_message}],
		"max_tokens": 512,
	}

	response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
	response.raise_for_status()
	data = response.json()
	return data["choices"][0]["message"]["content"]


st.title("Week 10 Final Project")
st.write("Hugging Face Inference Router setup test")

hf_token = get_hf_token()
if not hf_token:
	st.error(
		"Missing HF token. Add HF_TOKEN to .streamlit/secrets.toml (locally) or app Secrets (Streamlit Cloud)."
	)
	st.stop()

user_prompt = st.text_area("Prompt", "Say hello in one sentence.")

if st.button("Send"):
	with st.spinner("Calling model..."):
		try:
			assistant_text = chat_completion(user_prompt, hf_token)
			st.success("Request completed")
			st.write(assistant_text)
		except requests.HTTPError as http_err:
			status = http_err.response.status_code if http_err.response is not None else "unknown"
			detail = http_err.response.text if http_err.response is not None else str(http_err)
			st.error(f"API request failed (status {status}).")
			st.caption(detail)
		except requests.RequestException as req_err:
			st.error("Network/API error while contacting Hugging Face.")
			st.caption(str(req_err))
		except (KeyError, IndexError, TypeError, ValueError) as parse_err:
			st.error("Unexpected API response format.")
			st.caption(str(parse_err))

