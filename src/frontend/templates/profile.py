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