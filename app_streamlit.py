import streamlit as st
import pandas as pd
import os
import time
from streamlit_modal import Modal

# =====================
# Session State Init
# =====================
if "last_seen_phishing_ts" not in st.session_state:
    st.session_state.last_seen_phishing_ts = None

if "selected_email_id" not in st.session_state:
    st.session_state.selected_email_id = None

# new email modal
if "new_alert_email_id" not in st.session_state:
    st.session_state.new_alert_email_id = None

if "modal_shown_for_ts" not in st.session_state:
    st.session_state.modal_shown_for_ts = None

# confirm modal
if "pending_action" not in st.session_state:
    st.session_state.pending_action = None

if "pending_email_id" not in st.session_state:
    st.session_state.pending_email_id = None


# =====================
# Config
# =====================
st.set_page_config(
    page_title="Phishing Email Security Monitor",
    layout="wide"
)

# Hide Streamlit toolbar
st.markdown("""
<style>
div[data-testid="stToolbar"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

CSV_PATH = "email_events.csv"
REFRESH_SECONDS = 5

# auto refresh (new API, no warning)
st.query_params["_"] = str(int(time.time() / REFRESH_SECONDS))


# =====================
# Modals
# =====================
new_email_modal = Modal(
    title="🚨 New Phishing Email Detected",
    key="new-email-modal",
    padding=20,
    max_width=600
)

confirm_modal = Modal(
    title="⚠️ Confirm Security Action",
    key="confirm-modal",
    padding=20,
    max_width=500
)


# =====================
# Helpers
# =====================
@st.cache_data(ttl=REFRESH_SECONDS)
def load_emails():
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame()

    df = pd.read_csv(CSV_PATH)

    if "status" not in df.columns:
        df["status"] = "NEW"
    if "reasons" not in df.columns:
        df["reasons"] = ""
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def save_emails(df):
    df.to_csv(CSV_PATH, index=False)


def auto_promote_phishing(df):
    mask = (df["label"] == "PHISHING") & (df["status"] == "NEW")
    df.loc[mask, "status"] = "PHISHING_DETECTED"
    return df


# =====================
# Load + normalize
# =====================
df = load_emails()
df = auto_promote_phishing(df)
save_emails(df)


# =====================
# Detect new phishing
# =====================
phishing_df = df[df["label"] == "PHISHING"]

new_phishing_df = pd.DataFrame()
if not phishing_df.empty:
    if st.session_state.last_seen_phishing_ts is None:
        new_phishing_df = phishing_df
    else:
        new_phishing_df = phishing_df[
            phishing_df["timestamp"] > st.session_state.last_seen_phishing_ts
        ]

if not new_phishing_df.empty:
    newest = new_phishing_df.sort_values("timestamp", ascending=False).iloc[0]

    if st.session_state.modal_shown_for_ts != newest["timestamp"]:
        st.session_state.new_alert_email_id = newest["id"]
        st.session_state.modal_shown_for_ts = newest["timestamp"]
        new_email_modal.open()

new_phishing_count = len(new_phishing_df)


# =====================
# Header
# =====================
st.title("🛡️ Phishing Email Security Monitor")

if new_phishing_count > 0:
    st.warning(
        f"⚠️ {new_phishing_count} new phishing email(s) detected — review required"
    )


# =====================
# Email List
# =====================
st.subheader("📬 Detected Emails")

display_df = df.sort_values("timestamp", ascending=False).head(20)

options = []
option_to_id = {}

for _, row in display_df.iterrows():
    icon = "🚨" if row["label"] == "PHISHING" else "✅"
    text = f"{icon} {row['label']} | {row['subject']}"
    options.append(text)
    option_to_id[text] = row["id"]

if not options:
    st.info("No emails available")
    st.stop()

selected_option = st.radio(
    "Select an email to inspect",
    options,
    index=0
)

selected_id = option_to_id[selected_option]
selected_email = df[df["id"] == selected_id].iloc[0]

if selected_email["label"] == "PHISHING":
    ts = selected_email["timestamp"]
    if (
        st.session_state.last_seen_phishing_ts is None
        or ts > st.session_state.last_seen_phishing_ts
    ):
        st.session_state.last_seen_phishing_ts = ts


# =====================
# Active Threat Panel
# =====================
st.markdown("---")

if selected_email["label"] == "PHISHING":
    st.markdown(
        f"""
<div style="
    padding:30px;
    border-radius:12px;
    background-color:#fdecea;
    border:3px solid #d32f2f;
">
<h2 style="color:#b71c1c;margin-top:0;">🚨 ACTIVE SECURITY THREAT – ACTION REQUIRED</h2>

<b>From:</b> {selected_email['sender']}<br>
<b>To:</b> {selected_email['recipient']}<br>
<b>Subject:</b> {selected_email['subject']}<br><br>

<b>Detection Reasons:</b><br>
{selected_email['reasons']}<br><br>

<b>Status:</b> {selected_email['status']}
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    if c1.button("🚫 Quarantine Email", use_container_width=True):
        st.session_state.pending_action = "QUARANTINE"
        st.session_state.pending_email_id = selected_email["id"]
        confirm_modal.open()

    if c2.button("🧯 Block Sender", use_container_width=True):
        st.session_state.pending_action = "BLOCK"
        st.session_state.pending_email_id = selected_email["id"]
        confirm_modal.open()

    if c3.button("✅ Mark as Safe", use_container_width=True):
        st.session_state.pending_action = "MARK_SAFE"
        st.session_state.pending_email_id = selected_email["id"]
        confirm_modal.open()

else:
    st.success("✅ Selected email is benign. No action required.")


# =====================
# New Email Modal
# =====================
if new_email_modal.is_open():
    email = df[df["id"] == st.session_state.new_alert_email_id].iloc[0]

    with new_email_modal.container():
        st.markdown(
            f"""
**From:** {email['sender']}  
**To:** {email['recipient']}  
**Subject:** {email['subject']}  

**Detection Reasons:**  
{email['reasons']}
"""
        )

        st.divider()
        a, b, c = st.columns(3)

        if a.button("🚫 Quarantine"):
            st.session_state.pending_action = "QUARANTINE"
            st.session_state.pending_email_id = email["id"]
            new_email_modal.close()
            confirm_modal.open()

        if b.button("✅ Mark Safe"):
            st.session_state.pending_action = "MARK_SAFE"
            st.session_state.pending_email_id = email["id"]
            new_email_modal.close()
            confirm_modal.open()

        if c.button("❌ Close"):
            st.session_state.last_seen_phishing_ts = email["timestamp"]
            new_email_modal.close()


# =====================
# Confirm Modal
# =====================
if confirm_modal.is_open():
    email = df[df["id"] == st.session_state.pending_email_id].iloc[0]

    with confirm_modal.container():
        st.markdown(
            f"""
Are you sure you want to perform this action?

**Action:** {st.session_state.pending_action}  
**Email Subject:** {email['subject']}  
**From:** {email['sender']}
"""
        )

        st.divider()
        c1, c2 = st.columns(2)

        if c1.button("✅ Confirm"):
            if st.session_state.pending_action == "QUARANTINE":
                df.loc[df["id"] == email["id"], "status"] = "QUARANTINED"

            elif st.session_state.pending_action == "MARK_SAFE":
                df.loc[df["id"] == email["id"], ["status", "label"]] = ["RESOLVED", "BENIGN"]

            elif st.session_state.pending_action == "BLOCK":
                df.loc[df["id"] == email["id"], "status"] = "RESOLVED"

            save_emails(df)

            st.session_state.last_seen_phishing_ts = email["timestamp"]
            st.session_state.pending_action = None
            st.session_state.pending_email_id = None

            confirm_modal.close()
            st.rerun()

        if c2.button("❌ Cancel"):
            st.session_state.pending_action = None
            st.session_state.pending_email_id = None
            confirm_modal.close()
