import os
import uvicorn

from api.app import app

if __name__ == "__main__":
    """Run the Uvicorn server on the localhost at port 8080."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host=host, port=port)
