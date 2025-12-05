"""
Mistral 3 Lab - Streamlit interface til eksperimentering med Ministral 3 modeller
"""
import streamlit as st
import requests
import json
import time
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ollama API configuration
OLLAMA_BASE_URL = "http://localhost:11434"

# Available models
MODELS = {
    "ministral-3:3b": {"name": "Ministral 3B", "speed": "~361 tok/s", "desc": "Hurtigst"},
    "ministral-3:8b": {"name": "Ministral 8B", "speed": "~241 tok/s", "desc": "Balanceret"},
    "ministral-3:14b": {"name": "Ministral 14B", "speed": "~39 tok/s", "desc": "Bedst kvalitet"},
}

# Preset prompts
DEFAULT_PRESETS = {
    "Ingen": "",
    "Oversæt til engelsk": "Oversæt følgende tekst til engelsk:\n\n",
    "Forklar som til et barn": "Forklar følgende koncept som om du taler til et 10-årigt barn:\n\n",
    "Skriv Python kode": "Skriv Python kode der løser følgende opgave:\n\n",
    "Opsummer tekst": "Opsummer følgende tekst i 3-5 punkter:\n\n",
    "Ret grammatik": "Ret grammatik og stavefejl i følgende tekst:\n\n",
    "Generer idéer": "Generer 5 kreative idéer til:\n\n",
}


def check_ollama_running():
    """Check if Ollama is running"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_available_models():
    """Get list of available models from Ollama"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
    except:
        pass
    return []


def stream_chat(model, messages, options):
    """Stream chat response from Ollama API"""
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": options
    }

    try:
        with requests.post(url, json=payload, stream=True, timeout=300) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data:
                        yield data
                    if data.get("done", False):
                        yield data
                        break
    except requests.exceptions.RequestException as e:
        yield {"error": str(e)}


def chat_completion(model, messages, options):
    """Non-streaming chat completion"""
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options
    }

    start_time = time.time()
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        elapsed = time.time() - start_time
        data["elapsed_time"] = elapsed
        return data
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def encode_image(uploaded_file):
    """Encode uploaded image to base64 for Ollama vision API"""
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')


def stream_vision_chat(model, prompt, image_base64, options):
    """Stream vision chat response from Ollama API"""
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [image_base64]
        }],
        "stream": True,
        "options": options
    }

    try:
        with requests.post(url, json=payload, stream=True, timeout=300) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data:
                        yield data
                    if data.get("done", False):
                        yield data
                        break
    except requests.exceptions.RequestException as e:
        yield {"error": str(e)}


def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = {"prompt": 0, "completion": 0}
    if "custom_presets" not in st.session_state:
        st.session_state.custom_presets = {}
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = None
    if "vision_history" not in st.session_state:
        st.session_state.vision_history = []


def render_sidebar():
    """Render sidebar with settings"""
    with st.sidebar:
        st.title("⚙️ Settings")

        # Model selection
        st.subheader("Model")
        available_models = get_available_models()
        ministral_models = [m for m in MODELS.keys() if m in available_models]

        if not ministral_models:
            st.error("Ingen Ministral modeller fundet! Kør: ollama pull ministral-3:14b")
            model = list(MODELS.keys())[0]
        else:
            model = st.selectbox(
                "Vælg model",
                ministral_models,
                format_func=lambda x: f"{MODELS[x]['name']} ({MODELS[x]['desc']})"
            )

        # Model info
        if model in MODELS:
            st.caption(f"Hastighed: {MODELS[model]['speed']}")

        st.divider()

        # Generation settings
        st.subheader("Generation")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1,
                               help="Højere = mere kreativ, lavere = mere fokuseret")
        top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.05,
                         help="Nucleus sampling threshold")
        max_tokens = st.number_input("Max Tokens", 1, 4096, 1024,
                                    help="Maksimalt antal tokens i svar")
        repeat_penalty = st.slider("Repeat Penalty", 1.0, 2.0, 1.1, 0.1,
                                  help="Straffer gentagne ord")

        st.divider()

        # System prompt
        st.subheader("System Prompt")
        system_prompt = st.text_area(
            "System prompt",
            value="Du er en hjælpsom AI-assistent.",
            height=100,
            label_visibility="collapsed"
        )

        st.divider()

        # Preset prompts
        st.subheader("Preset Prompts")
        all_presets = {**DEFAULT_PRESETS, **st.session_state.custom_presets}
        selected_preset = st.selectbox("Vælg preset", list(all_presets.keys()))

        # Add custom preset
        with st.expander("Tilføj custom preset"):
            new_preset_name = st.text_input("Navn")
            new_preset_text = st.text_area("Tekst", height=100)
            if st.button("Gem preset") and new_preset_name and new_preset_text:
                st.session_state.custom_presets[new_preset_name] = new_preset_text
                st.success(f"Preset '{new_preset_name}' gemt!")
                st.rerun()

        st.divider()

        # Token stats
        st.subheader("📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prompt", st.session_state.total_tokens["prompt"])
        with col2:
            st.metric("Completion", st.session_state.total_tokens["completion"])
        st.metric("Total", sum(st.session_state.total_tokens.values()))

        # Clear chat button
        if st.button("🗑️ Ryd chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_tokens = {"prompt": 0, "completion": 0}
            st.rerun()

    options = {
        "temperature": temperature,
        "top_p": top_p,
        "num_predict": max_tokens,
        "repeat_penalty": repeat_penalty,
    }

    return model, system_prompt, options, all_presets.get(selected_preset, "")


def render_chat_tab(model, system_prompt, options, preset_text):
    """Render the chat interface tab"""
    st.header("💬 Chat")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "stats" in message:
                stats = message["stats"]
                st.caption(
                    f"📊 {stats.get('tokens', 'N/A')} tokens | "
                    f"⚡ {stats.get('speed', 'N/A')} tok/s | "
                    f"⏱️ {stats.get('time', 'N/A')}s"
                )

    # Chat input
    if prompt := st.chat_input("Skriv din besked..."):
        # Apply preset if selected
        if preset_text and not prompt.startswith(preset_text):
            full_prompt = preset_text + prompt
        else:
            full_prompt = prompt

        # Add user message
        st.session_state.messages.append({"role": "user", "content": full_prompt})
        with st.chat_message("user"):
            st.markdown(full_prompt)

        # Prepare messages for API
        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend([
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ])

        # Stream response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            stats_placeholder = st.empty()

            full_response = ""
            prompt_tokens = 0
            completion_tokens = 0
            start_time = time.time()

            for chunk in stream_chat(model, api_messages, options):
                if "error" in chunk:
                    st.error(f"Fejl: {chunk['error']}")
                    break

                if "message" in chunk:
                    content = chunk["message"].get("content", "")
                    full_response += content
                    message_placeholder.markdown(full_response + "▌")

                if chunk.get("done", False):
                    prompt_tokens = chunk.get("prompt_eval_count", 0)
                    completion_tokens = chunk.get("eval_count", 0)

            elapsed = time.time() - start_time
            speed = completion_tokens / elapsed if elapsed > 0 else 0

            message_placeholder.markdown(full_response)
            stats_placeholder.caption(
                f"📊 {completion_tokens} tokens | "
                f"⚡ {speed:.1f} tok/s | "
                f"⏱️ {elapsed:.1f}s"
            )

            # Update session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "stats": {
                    "tokens": completion_tokens,
                    "speed": f"{speed:.1f}",
                    "time": f"{elapsed:.1f}"
                }
            })
            st.session_state.total_tokens["prompt"] += prompt_tokens
            st.session_state.total_tokens["completion"] += completion_tokens


def render_comparison_tab(system_prompt, options):
    """Render the model comparison tab"""
    st.header("🔄 Model Sammenligning")
    st.caption("Send samme prompt til alle modeller og sammenlign resultaterne")

    # Get available ministral models
    available_models = get_available_models()
    ministral_models = [m for m in MODELS.keys() if m in available_models]

    if len(ministral_models) < 2:
        st.warning("Du skal have mindst 2 Ministral modeller installeret for sammenligning.")
        return

    # Comparison prompt input
    comparison_prompt = st.text_area(
        "Prompt til sammenligning",
        height=100,
        placeholder="Skriv en prompt der sendes til alle modeller..."
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run_comparison = st.button("🚀 Kør sammenligning", use_container_width=True)

    if run_comparison and comparison_prompt:
        st.divider()

        # Create columns for each model
        cols = st.columns(len(ministral_models))

        results = {}

        # Run comparisons in parallel
        def run_model(model):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": comparison_prompt}
            ]
            return model, chat_completion(model, messages, options)

        with st.spinner("Kører alle modeller..."):
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(run_model, m): m for m in ministral_models}
                for future in as_completed(futures):
                    model, result = future.result()
                    results[model] = result

        # Display results
        for idx, model in enumerate(ministral_models):
            with cols[idx]:
                st.subheader(MODELS[model]["name"])

                result = results.get(model, {})

                if "error" in result:
                    st.error(f"Fejl: {result['error']}")
                else:
                    # Stats
                    elapsed = result.get("elapsed_time", 0)
                    eval_count = result.get("eval_count", 0)
                    speed = eval_count / elapsed if elapsed > 0 else 0

                    st.caption(
                        f"⏱️ {elapsed:.1f}s | "
                        f"📊 {eval_count} tok | "
                        f"⚡ {speed:.1f} tok/s"
                    )

                    # Response
                    response = result.get("message", {}).get("content", "Intet svar")
                    st.markdown(response)

        # Store results
        st.session_state.comparison_results = results


def render_vision_tab(model, options):
    """Render the vision/multimodal tab"""
    st.header("🖼️ Vision")
    st.caption("Upload et billede og stil spørgsmål om det - Ministral 3 modellerne er multimodale!")

    # Image upload
    uploaded_file = st.file_uploader(
        "Upload billede",
        type=["png", "jpg", "jpeg", "webp"],
        help="Understøttede formater: PNG, JPG, JPEG, WEBP"
    )

    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(uploaded_file, caption="Uploaded billede", use_container_width=True)

        with col2:
            # Preset vision prompts
            vision_presets = {
                "Custom": "",
                "Beskriv billedet": "Beskriv hvad du ser på dette billede i detaljer.",
                "Identificer objekter": "Identificer og list alle objekter du kan se på billedet.",
                "Læs tekst": "Læs og transskriber al tekst der er synlig på billedet.",
                "Analyser stemning": "Analyser stemningen og atmosfæren i dette billede.",
            }

            selected_vision_preset = st.selectbox(
                "Vælg prompt type",
                list(vision_presets.keys())
            )

            default_prompt = vision_presets.get(selected_vision_preset, "")

            vision_prompt = st.text_area(
                "Dit spørgsmål om billedet",
                value=default_prompt,
                height=100,
                placeholder="Skriv dit spørgsmål om billedet her..."
            )

            analyze_button = st.button("🔍 Analyser billede", use_container_width=True)

        if analyze_button and vision_prompt:
            st.divider()

            # Encode image
            image_base64 = encode_image(uploaded_file)

            # Stream response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                stats_placeholder = st.empty()

                full_response = ""
                completion_tokens = 0
                start_time = time.time()

                for chunk in stream_vision_chat(model, vision_prompt, image_base64, options):
                    if "error" in chunk:
                        st.error(f"Fejl: {chunk['error']}")
                        break

                    if "message" in chunk:
                        content = chunk["message"].get("content", "")
                        full_response += content
                        message_placeholder.markdown(full_response + "▌")

                    if chunk.get("done", False):
                        completion_tokens = chunk.get("eval_count", 0)

                elapsed = time.time() - start_time
                speed = completion_tokens / elapsed if elapsed > 0 else 0

                message_placeholder.markdown(full_response)
                stats_placeholder.caption(
                    f"📊 {completion_tokens} tokens | "
                    f"⚡ {speed:.1f} tok/s | "
                    f"⏱️ {elapsed:.1f}s"
                )

                # Save to history
                st.session_state.vision_history.append({
                    "image_name": uploaded_file.name,
                    "prompt": vision_prompt,
                    "response": full_response,
                    "model": model,
                    "stats": {
                        "tokens": completion_tokens,
                        "speed": f"{speed:.1f}",
                        "time": f"{elapsed:.1f}"
                    }
                })

                # Update token counter
                st.session_state.total_tokens["completion"] += completion_tokens

    # Display vision history
    if st.session_state.vision_history:
        st.divider()
        st.subheader("📜 Vision historik")
        for idx, entry in enumerate(reversed(st.session_state.vision_history[-5:])):
            with st.expander(f"🖼️ {entry['image_name']} - {entry['model']}"):
                st.markdown(f"**Prompt:** {entry['prompt']}")
                st.markdown(f"**Svar:** {entry['response']}")
                st.caption(
                    f"📊 {entry['stats']['tokens']} tokens | "
                    f"⚡ {entry['stats']['speed']} tok/s | "
                    f"⏱️ {entry['stats']['time']}s"
                )


def main():
    """Main application"""
    st.set_page_config(
        page_title="Mistral 3 Lab",
        page_icon="🔬",
        layout="wide"
    )

    # Initialize session state
    init_session_state()

    # Check if Ollama is running
    if not check_ollama_running():
        st.error("⚠️ Ollama kører ikke! Start Ollama med: `ollama serve`")
        st.stop()

    # Render sidebar and get settings
    model, system_prompt, options, preset_text = render_sidebar()

    # Main content with tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🖼️ Vision", "🔄 Sammenligning"])

    with tab1:
        render_chat_tab(model, system_prompt, options, preset_text)

    with tab2:
        render_vision_tab(model, options)

    with tab3:
        render_comparison_tab(system_prompt, options)


if __name__ == "__main__":
    main()
