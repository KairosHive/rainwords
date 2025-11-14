import uvicorn
import webbrowser
import time
from threading import Thread


def open_browser_when_ready(url: str, delay: float = 10.0):
    """
    Opens the browser after a short delay.
    Delay is needed because Uvicorn takes time to initialize.
    """
    def _open():
        time.sleep(delay)
        webbrowser.open(url)
    Thread(target=_open, daemon=True).start()


def main():
    """
    Entry point for the `rainwords` CLI command.
    Starts the backend AND automatically opens the UI in the browser.
    """
    url = "http://127.0.0.1:8000"

    # auto-open browser shortly after startup
    open_browser_when_ready(url)

    uvicorn.run(
        "rainwords.main:app",   # app import string
        host="127.0.0.1",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
