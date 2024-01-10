import os
import uvicorn

from app import ChatModel, create_app


def main(): 
    chat_model = ChatModel()
    app = create_app(chat_model)
    uvicorn.run(app, host="0.0.0.0", port=12345, workers=1)


if __name__ == '__main__': 

    main()