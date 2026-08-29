import os
import streamlit as st
import ee

def _ee_credentials_from_file():
    """Load OAuth2 credentials from the earthengine credentials file (no gcloud needed)."""
    import pathlib, json as _json
    cred_file = pathlib.Path.home() / ".config" / "earthengine" / "credentials"
    if not cred_file.exists():
        return None
    try:
        import google.oauth2.credentials
        data = _json.loads(cred_file.read_text())
        return google.oauth2.credentials.Credentials(
            token=None,
            refresh_token=data["refresh_token"],
            token_uri="https://oauth2.googleapis.com/token",
            client_id=data["client_id"],
            client_secret=data["client_secret"],
        )
    except Exception:
        return None


def _is_cloud_deployment() -> bool:
    """Return True when running on Streamlit Cloud or any headless server."""
    cloud_signals = [
        os.environ.get("STREAMLIT_SHARING_MODE"),
        os.environ.get("IS_CLOUD_ENV"),
        os.environ.get("STREAMLIT_SERVER_HEADLESS"),
    ]
    return any(v for v in cloud_signals)


_STREAMLIT_CLOUD_SETUP = """
**Production deployment detected — service account credentials required.**

Add the following to your app's **Streamlit Cloud → Settings → Secrets**:

```toml
[gee]
credentials = '''
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "...",
  "private_key": "-----BEGIN RSA PRIVATE KEY-----\\n...\\n-----END RSA PRIVATE KEY-----\\n",
  "client_email": "your-sa@your-project.iam.gserviceaccount.com",
  ...
}
'''
```

Steps:
1. Go to [Google Cloud Console](https://console.cloud.google.com/) → IAM & Admin → Service Accounts
2. Create a service account and grant it the **Earth Engine Resource Viewer** role
3. Create a JSON key and paste the full JSON as the `credentials` value above
4. Redeploy the app
"""


def init_earth_engine(project_id: str) -> tuple[bool, str]:
    import json as _json

    # 1. Service-account path — Streamlit Cloud deployment via st.secrets
    _sa_creds = st.secrets.get("gee", {}).get("credentials", "").strip()
    if _sa_creds:
        try:
            cred_dict = _json.loads(_sa_creds)
            credentials = ee.ServiceAccountCredentials(
                email=cred_dict["client_email"],
                key_data=_sa_creds,
            )
            ee.Initialize(credentials=credentials, project=project_id)
            return True, "Earth Engine initialized via service account."
        except Exception as e:
            return False, f"Service account auth failed: {e}"

    # On Streamlit Cloud there is no home directory with credentials and no
    # browser to complete OAuth — skip straight to a clear setup message.
    if _is_cloud_deployment():
        return False, _STREAMLIT_CLOUD_SETUP

    # 2. Explicit credentials file — written by `earthengine authenticate` (no gcloud needed)
    local_creds = _ee_credentials_from_file()
    if local_creds is not None:
        try:
            ee.Initialize(credentials=local_creds, project=project_id)
            return True, "Earth Engine initialized."
        except Exception:
            pass  # fall through to browser auth

    # 3. Browser OAuth — only attempted on local machines
    try:
        ee.Authenticate(auth_mode="localhost", force=False)
        local_creds = _ee_credentials_from_file()
        if local_creds is not None:
            ee.Initialize(credentials=local_creds, project=project_id)
            return True, "Earth Engine authenticated and initialized."
        ee.Initialize(project=project_id)
        return True, "Earth Engine authenticated and initialized."
    except OSError as e:
        if getattr(e, "errno", None) == 98 or "Address already in use" in str(e):
            return False, (
                "Could not start the local OAuth server (port already in use). "
                "Run `earthengine authenticate --auth_mode notebook` in a terminal, "
                "then restart the app."
            )
        return False, f"Authentication failed: {e}"
    except Exception as e:
        return False, (
            f"Authentication failed ({e}). "
            "Run `earthengine authenticate` in a terminal and restart the app."
        )

