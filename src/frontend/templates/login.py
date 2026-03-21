from __future__ import annotations
import streamlit as st
from pathlib import Path


def render_login_page() -> None:
    """
    Renders the full-screen login page.
    Called from app.py when st.user.is_logged_in is False.

    Uses Streamlit's native st.login() which handles the entire
    Google OIDC flow — redirect, callback, cookie — automatically.
    No manual OAuth session management needed.
    """

    _css = (Path(__file__).parent / "login.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

    # ── Logo ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="login-logo">🧮</div>', unsafe_allow_html=True)

    # ── Card ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="login-card">', unsafe_allow_html=True)

    st.markdown('<div class="login-title">JEE Math Tutor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="login-subtitle">'
        'AI-powered step-by-step solutions,<br>'
        'personalised to how you learn.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Feature pills
    st.markdown("""
    <div class="feature-row">
        <span class="feature-pill">📐 All JEE Topics</span>
        <span class="feature-pill">🧠 Memory Across Sessions</span>
        <span class="feature-pill">🎬 Visual Explanations</span>
        <span class="feature-pill">📄 Upload Study Material</span>
    </div>
    """, unsafe_allow_html=True)

    # Google login button — st.login() handles the entire OIDC flow
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        if st.button("🔵  Continue with Google", use_container_width=True):
            st.login("google")

    # Divider
    st.markdown('<div class="login-divider">secure sign-in</div>', unsafe_allow_html=True)

    # Trust badges
    st.markdown("""
    <div class="trust-row">
        <span class="trust-item">🔒 Google OAuth 2.0</span>
        <span class="trust-item">🛡️ No password stored</span>
        <span class="trust-item">☁️ Encrypted session</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown(
        '<div class="login-footer">By signing in you agree to use this tool for educational purposes.</div>',
        unsafe_allow_html=True,
    )