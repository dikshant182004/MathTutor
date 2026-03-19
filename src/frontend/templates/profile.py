from frontend.templates import html, time
from frontend import Optional

def build_profile_card(
    name:             str,
    email:            str,
    problems_solved:  int  = 0,
    last_login:       Optional[float] = None,
    member_since:     Optional[float] = None,
) -> str:
    """
    Builds the HTML profile card shown inside the top-right popover.

    Args:
        name            : display name from Google OAuth
        email           : email from Google OAuth
        problems_solved : total_problems_solved from Redis user hash
        last_login      : Unix timestamp of last login (from Redis)
        member_since    : Unix timestamp of account creation (from Redis)

    Returns raw HTML string.

    Called in new_app.py after login — data comes from get_user_profile()
    in db_utils.py.
    """
    n = html.escape(name)
    e = html.escape(email)

    last_login_str = (
        time.strftime("%d %b %Y, %H:%M", time.localtime(last_login))
        if last_login else "—"
    )
    member_since_str = (
        time.strftime("%d %b %Y", time.localtime(member_since))
        if member_since else "—"
    )

    return f"""
<div class="profile-card">
    <div class="name">{n}</div>
    <div class="email">{e}</div>
    <div class="stat">
        ✅ Problems solved: <span>{problems_solved}</span>
    </div>
    <div class="stat">
        🕐 Last login: <span>{last_login_str}</span>
    </div>
    <div class="stat">
        📅 Member since: <span>{member_since_str}</span>
    </div>
</div>
"""