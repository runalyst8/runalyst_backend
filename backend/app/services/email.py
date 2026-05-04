import resend
from app.core.config import settings


def send_verification_email(to_email: str, code: str) -> None:
    resend.api_key = settings.RESEND_API_KEY
    resend.Emails.send({
        "from": settings.RESEND_FROM_EMAIL,
        "to": to_email,
        "subject": "Verify your Runalyst account",
        "html": f"""
        <div style="font-family:sans-serif;max-width:480px;margin:auto;padding:32px">
            <h2 style="color:#1E293B">Verify your email</h2>
            <p style="color:#475569">Use the code below to verify your Runalyst account.
            It expires in 10 minutes.</p>
            <div style="font-size:36px;font-weight:800;letter-spacing:8px;
                        color:#2563EB;text-align:center;padding:24px 0">
                {code}
            </div>
            <p style="color:#94A3B8;font-size:13px">
                If you did not create an account, you can ignore this email.
            </p>
        </div>
        """,
    })
