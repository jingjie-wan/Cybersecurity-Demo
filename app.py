# app.py — Phishing Email Security Dashboard (SOC-style)
# Place df_result.csv in the same folder. Expected columns:
# ['text_combined','label','timestamp','recipient','attachment','prediction','response_action']

import pandas as pd
import numpy as np
from datetime import timedelta
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from urllib.parse import quote

st.set_page_config(page_title="Phishing Email Security Dashboard", layout="wide")
# ---------- Global CSS (bigger KPI + subtle UI) ----------
st.markdown("""
<style>
a[data-testid="stAppViewSource"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
.main .block-container { padding-top: 1.0rem; padding-bottom: 1.6rem; }
h1 { font-size: 1.8rem; margin: .2rem 0 .4rem 0; }
h2, h3 { font-size: 1.2rem; margin: .4rem 0 .4rem 0; font-weight: 700; }
div[data-testid="stMetricValue"] { font-size: 44px !important; line-height: 1.05; }
div[data-testid="stMetricLabel"] { font-size: 14px !important; }

/* Pills (Darktrace-style) */
.time-pills-wrap { display:flex; justify-content:flex-end; margin-top:.25rem; }
.time-pills-wrap .stRadio > div { gap: .5rem; }
.time-pills-wrap [role="radiogroup"] {
  display:flex; gap:.5rem; padding:.2rem; background:#1f2937;
  border:1px solid #2f3b52; border-radius:999px;
}
.time-pills-wrap [role="radio"] {
  padding:.35rem .8rem; border-radius:999px; cursor:pointer;
  color:#cbd5e1; border:1px solid transparent;
}
.time-pills-wrap [role="radio"]:hover { background:#263348; color:#e2e8f0; }
.time-pills-wrap [aria-checked="true"] {
  background:#2f3b52; color:#fff; border-color:#3b4a66;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.08);
}

/* Pretty mail buttons in tables */
a.mail-btn {
  background: #2b6cb0; color: white; padding: 6px 10px; border-radius: 8px;
  text-decoration: none; font-weight: 600;
}
a.mail-btn:hover { filter: brightness(1.15); }

/* Color chips for response actions (legend) */
span.badge { padding: 3px 8px; border-radius: 8px; color: #fff; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# ---------- Load & prepare data ----------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["timestamp"].dt.date
    # hygiene
    for c in ["text_combined","recipient","attachment","response_action"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    for c in ["label","prediction"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    # derived
    df["is_threat"] = (df.get("prediction", 0) == 1).astype(int)   # treat prediction as system detection
    return df

try:
    df = load_data("df_result_demo.csv")
except Exception as e:
    st.error("Haven't found df_result_demo.csv. Please put the .csv file in the same directory as app.py.")
    st.stop()

# ---------- Sidebar filters (keep only SOC-style controls) ----------
st.sidebar.header("Filters")

min_date = pd.to_datetime(df['date']).min()
max_date = pd.to_datetime(df['date']).max()
default_start = max_date - timedelta(days=14) if pd.notnull(max_date) else min_date

date_range = st.sidebar.date_input(
    "Date range", (default_start, max_date),
    min_value=min_date, max_value=max_date
)
if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# ---------- Header ----------
hl, hr = st.columns([6, 4])

with hl:
    st.title("Phishing Email Security Dashboard")

with hr:
    st.markdown('<div class="time-pills-wrap">', unsafe_allow_html=True)
    time_override = st.radio(
        label="",  # hide label entirely
        options=["24 Hours", "Week", "Month", "Custom"],
        index=3,               # default to Custom uses sidebar dates
        horizontal=True,
        key="time_override_radio",
        label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)

# --- Effective time window (override if pills selected) ---

# anchor = now (or last data point if available)
anchor = pd.Timestamp.now() if df["timestamp"].isna().all() else pd.to_datetime(df["timestamp"].max())

# normalize anchor to the end of its day
anchor = anchor.replace(hour=23, minute=59, second=59)

if time_override == "24 Hours":
    base_start = (anchor - pd.Timedelta(hours=24) + pd.Timedelta(hours=24)).replace(hour=0, minute=0, second=0)
    eff_start, eff_end = base_start, anchor

elif time_override == "Week":
    base_start = (anchor - pd.Timedelta(days=7) + pd.Timedelta(hours=24)).replace(hour=0, minute=0, second=0)
    eff_start, eff_end = base_start, anchor

elif time_override == "Month":
    base_start = (anchor - pd.DateOffset(months=1) + pd.Timedelta(hours=24)).replace(hour=0, minute=0, second=0)
    eff_start, eff_end = base_start, anchor

else:  # fallback to sidebar range
    eff_start = pd.to_datetime(start_date).replace(hour=0, minute=0, second=0)
    eff_end   = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)

attachments = sorted(df['attachment'].dropna().unique().tolist()) if 'attachment' in df.columns else []
selected_attachments = st.sidebar.multiselect("Attachment type", attachments, default=attachments)

recipient_query = st.sidebar.text_input("Recipient contains (optional)")

threat_filter = st.sidebar.selectbox("Threat filter", ["All", "Threats only", "Safe only"], index=0)

# Apply filters
# f = df.copy()
# f = f[(f['date'] >= pd.to_datetime(start_date)) & (f['date'] <= pd.to_datetime(end_date))]
f = df.copy()
f = f[(f["timestamp"] >= eff_start) & (f["timestamp"] <= eff_end)]

if selected_attachments and 'attachment' in f.columns:
    f = f[f['attachment'].isin(selected_attachments)]
if recipient_query:
    f = f[f['recipient'].str.contains(recipient_query, case=False, na=False)]
if threat_filter == "Threats only":
    f = f[f['is_threat'] == 1]
elif threat_filter == "Safe only":
    f = f[f['is_threat'] == 0]

# Time caption
st.caption(f"Showing: {time_override}  •  {eff_start:%Y-%m-%d %H:%M} → {eff_end:%Y-%m-%d %H:%M}")
# ---------- KPI Row ----------
col1, col2, col3 = st.columns([1,1,1])

total_emails = len(f)
threats = int(f["is_threat"].sum())
quarantined = int(f["response_action"].str.contains("quarantined", case=False, na=False).sum()) if "response_action" in f else 0

col1.metric("Emails Scanned", f"{total_emails:,}")
col2.metric("Threats Detected", f"{threats:,}", f"{(threats/max(total_emails,1))*100:.1f}%")
col3.metric("Quarantined Emails", f"{quarantined:,}")

st.markdown("---")

# ---------- Combined Time Series with markers ----------
ts = f[['timestamp','is_threat']].dropna().copy()
# 自动判断：如果时间窗口 ≤ 48 小时，用小时聚合，否则按天聚合
window_hours = (eff_end - eff_start).total_seconds() / 3600.0

if window_hours <= 48:
    ts = ts.set_index('timestamp').sort_index()
    daily = ts.resample('H').agg(
        emails_scanned=('is_threat', 'count'),
        threats_detected=('is_threat', 'sum')
    ).reset_index().rename(columns={'timestamp': 'date'})
else:
    ts['date'] = ts['timestamp'].dt.date
    daily = ts.groupby('date').agg(
        emails_scanned=('is_threat','count'),
        threats_detected=('is_threat','sum')
    ).reset_index()


# amplify differences
# amplify differences + keep integers
daily['emails_scanned'] = (daily['emails_scanned'] * np.random.uniform(0.8, 1.2, len(daily))).round().astype(int)
daily['threats_detected'] = (daily['threats_detected'] * np.random.uniform(0.5, 2.5, len(daily))).round().astype(int)


if not daily.empty:
    y_max = max(daily['emails_scanned'].max(), daily['threats_detected'].max())
    if y_max > 0:
        y95 = float(np.percentile(pd.concat([daily['emails_scanned'], daily['threats_detected']]), 95))
        upper = max(10, y95 * 1.15)  # focus the y-range to be more “concentrated”
    else:
        upper = 10

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily['date'], y=daily['emails_scanned'],
        mode='lines+markers', name='Emails Scanned'
    ))
    fig.add_trace(go.Scatter(
        x=daily['date'], y=daily['threats_detected'],
        mode='lines+markers', name='Threats Detected'
    ))

    # 根据聚合方式动态调整标题文字
    if window_hours <= 48:
        chart_title = "Hourly Emails & Threats"
    else:
        chart_title = "Daily Emails & Threats"

    fig.update_layout(
        title=chart_title,
        margin=dict(l=10,r=10,t=60,b=10),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

    fig.update_yaxes(range=[0, upper])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No time-series data after filters.")

st.markdown("---")

# ---------- Donut charts: Response Actions & Attachments ----------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Response Actions")
    if 'response_action' in f.columns and not f['response_action'].dropna().empty:
        counts = f['response_action'].value_counts().reset_index()
        counts.columns = ['response_action','count']
        fig_ra = px.pie(counts, names='response_action', values='count', hole=0.55)
        fig_ra.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=320, showlegend=True)
        st.plotly_chart(fig_ra, use_container_width=True)
    else:
        st.write("—")

with c2:
    st.subheader("Attachments")
    if 'attachment' in f.columns and not f['attachment'].dropna().empty:
        counts = f['attachment'].value_counts().reset_index()
        counts.columns = ['attachment','count']
        fig_att = px.pie(counts, names='attachment', values='count', hole=0.55)
        fig_att.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=320, showlegend=True)
        st.plotly_chart(fig_att, use_container_width=True)
    else:
        st.write("—")

st.markdown("---")

# ---------- Top Targeted Recipients (table + email button) ----------
st.subheader("Top Targeted Recipients (Threats)")
if 'recipient' in f.columns:
    top = f[f['is_threat']==1]['recipient'].value_counts().head(15).reset_index()
    top.columns = ['recipient','count']
    if not top.empty:
        # Build HTML table with an "Email" button
        rows_html = []
        for _, r in top.iterrows():
            rcpt = r['recipient']
            cnt = int(r['count'])
            # Build a mailto with prefilled subject & body
            subject = quote("Security Alert: Suspicious Email Activity")
            body = quote(f"Hi {rcpt.split('@')[0]},\n\nWe detected suspicious emails targeting your mailbox in the selected period. Please be cautious with unexpected attachments and links.\n\n— Security Team")
            mail_link = f"mailto:{rcpt}?subject={subject}&body={body}"
            rows_html.append(
                f"<tr><td>{rcpt}</td><td style='text-align:right'>{cnt}</td>"
                f"<td style='text-align:right'><a class='mail-btn' href='{mail_link}'>Email</a></td></tr>"
            )
        html = f"""
        <table style="width:100%; border-collapse:collapse;">
          <thead>
            <tr style="text-align:left; border-bottom:1px solid #ddd;">
              <th>Recipient</th><th style="text-align:right">Threat Count</th><th style="text-align:right">Action</th>
            </tr>
          </thead>
          <tbody>{''.join(rows_html)}</tbody>
        </table>
        """
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.write("—")
else:
    st.write("—")

st.markdown("---")

# ---------- Recent Threats (row-colored table) ----------
st.subheader("Most Recent Threats (Last 25)")
# mapping colors by response_action
color_map = {
    "Email quarantined": "#22543d",    # green-ish
    "Sender blocked": "#6b46c1",       # purple
    "Alert sent to IT": "#2c5282",     # blue
    "Suspicious link removed": "#9b2c2c",  # red
}

cols = ['timestamp','recipient','attachment','response_action','text_combined']
cols = [c for c in cols if c in f.columns]
recent = f.sort_values('timestamp', ascending=False)
recent = recent[recent['is_threat']==1].head(25).copy()
if 'text_combined' in recent.columns:
    recent['text_preview'] = recent['text_combined'].apply(lambda x: x[:160] + ("…" if isinstance(x,str) and len(x)>160 else ""))
    show_cols = [c for c in cols if c != 'text_combined'] + ['text_preview']
else:
    show_cols = cols

if not recent.empty:
    def highlight_row(row):
        ra = str(row.get('response_action', ''))
        # pick color if matched, else neutral
        bg = color_map.get(ra, "#2d3748" if st.get_option("theme.base")=="dark" else "#717275")
        return [f'background-color: {bg}; color: white;' for _ in row]

    styled = recent[show_cols].reset_index(drop=True).style.apply(highlight_row, axis=1)
    st.dataframe(styled, use_container_width=True)
else:
    st.write("—")

# ---------- Footer ----------
st.caption("Tip: Use the sidebar to adjust date range, filter by attachment, or search recipients. Toggle prediction filter to focus on phishing.")
